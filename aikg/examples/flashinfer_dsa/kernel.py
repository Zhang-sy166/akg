import triton
import triton.language as tl
import torch

@triton.jit
def dsa_indexer_kernel(
    q_ptr,              # [B, H, D] float32
    K_all_ptr,          # [NP, P, D] float32
    weights_ptr,        # [B, H] float32
    seq_lens_ptr,       # [B] int32
    block_table_ptr,    # [B, MNP] int32
    final_scores_ptr,   # [B, MAX_SEQ_LEN] float32
    global_indices_ptr, # [B, MAX_SEQ_LEN] int32

    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    NP: tl.constexpr,
    MNP: tl.constexpr,

    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_K_np: tl.constexpr,
    stride_K_p: tl.constexpr,
    stride_K_d: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_bt_b: tl.constexpr,
    stride_bt_np: tl.constexpr,
    stride_fs_b: tl.constexpr,
    stride_fs_s: tl.constexpr,
    stride_gi_b: tl.constexpr,
    stride_gi_s: tl.constexpr,

    BLOCK_S: tl.constexpr,   # token block size
    BLOCK_D: tl.constexpr,   # head_dim block size
):
    # 每个 program 处理一个 batch b
    b = tl.program_id(0)
    if b >= B:
        return

    # 加载该 batch 的 seq_len
    seq_len = tl.load(seq_lens_ptr + b)
    if seq_len <= 0:
        return  # Python端已经用zeros和full初始化好了，直接返回即可

    # 该序列需要的页数
    num_pages = (seq_len + P - 1) // P

    # 指向本 batch 的输出
    fs_b_base = final_scores_ptr + b * stride_fs_b
    gi_b_base = global_indices_ptr + b * stride_gi_b

    # 主循环：分块处理 tokens，块大小 BLOCK_S
    for s_start in range(0, seq_len, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
        mask_s = offs_s < seq_len

        # 计算 page 和 offset
        page_idx = offs_s // P                     # [BLOCK_S]
        offset   = offs_s % P                      # [BLOCK_S]
        mask_page = page_idx < num_pages

        # 取出 global_page：从 block_table[b, page_idx]
        bt_row = block_table_ptr + b * stride_bt_b
        global_page = tl.load(
            bt_row + page_idx * stride_bt_np,
            mask=mask_page & mask_s,
            other=0
        )  # [BLOCK_S], int32

        # 计算 global token index: page * P + offset
        global_token_idx = global_page * P + offset  # [BLOCK_S]
        
        # 存到 global_indices[b, offs_s]
        gi_ptr = gi_b_base + offs_s * stride_gi_s
        tl.store(gi_ptr, global_token_idx, mask=mask_s)

        # 准备累积该 block 的 final_scores_tile
        fs_tile = tl.zeros([BLOCK_S], dtype=tl.float32)

        # 对每个 head h 累加：scores_h = relu(q_h @ k_block) * w_h
        for h in range(0, H):
            dot_h = tl.zeros([BLOCK_S], dtype=tl.float32)
            # 加载当前头对应的权重
            weight_cur_head = tl.load(weights_ptr + b * stride_w_b + h * stride_w_h)

            # 内层按 D 维分块进行 dot
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
                mask_d = offs_d < D

                # q[b, h, d]
                q_base = q_ptr + b * stride_q_b + h * stride_q_h
                q_vals = tl.load(
                    q_base + offs_d * stride_q_d,
                    mask=mask_d,
                    other=0.0
                )  # [BLOCK_D]

                # K_all[global_page, offset, d]
                k_ptrs = (
                    K_all_ptr
                    + global_page[:, None] * stride_K_np
                    + offset[:, None] * stride_K_p
                    + offs_d[None, :] * stride_K_d
                )  # [BLOCK_S, BLOCK_D]

                k_vals = tl.load(
                    k_ptrs,
                    mask=mask_s[:, None] & mask_d[None, :],
                    other=0.0
                )  # [BLOCK_S, BLOCK_D]

                # 累加 dot
                partial = tl.sum(k_vals * q_vals[None, :], axis=1)  # [BLOCK_S]
                dot_h += partial

            # ReLU & multiply weight
            dot_h = tl.maximum(dot_h, 0.0)
            dot_h = dot_h * weight_cur_head

            fs_tile += dot_h  # 累加到最终 scores

        # 把该块 scores 写回显存
        fs_ptr = fs_b_base + offs_s * stride_fs_s
        tl.store(fs_ptr, fs_tile, mask=mask_s)


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    device = q_index_fp8.device
    B, H, D = q_index_fp8.shape
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape

    assert H == 64
    assert D == 128
    assert page_size == 64

    topk = 2048
    K_val = topk
    MNP = block_table.shape[1]
    
    def dequant_fp8_kv_cache(k_index_cache_fp8):
        k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
        num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4  
        
        kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
        
        fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
        fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
        fp8_float = fp8_tensor.to(torch.float32)
        
        scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
        scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32) 
        
        return fp8_float * scale

    # 解量化
    q = q_index_fp8.to(torch.float32)                  
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  

    # 计算 max_seq_len，向上取整到 page_size 的倍数
    max_seq_len = int(seq_lens.max().item())
    max_seq_len = ((max_seq_len + page_size - 1) // page_size) * page_size
    max_seq_len = triton.next_power_of_2(max_seq_len)

    # 【这里已经在 Python 侧做好了零值和 -1 初始化，内核不用管了】
    final_scores = torch.zeros(
        (B, max_seq_len), device=device, dtype=torch.float32
    )
    global_indices = torch.full(
        (B, max_seq_len), -1, device=device, dtype=torch.int32
    )

    stride_q_b, stride_q_h, stride_q_d = q.stride()
    stride_K_np, stride_K_p, stride_K_d = K_all.stride()
    stride_w_b, stride_w_h = weights.stride()
    stride_bt_b, stride_bt_np = block_table.stride()
    stride_fs_b, stride_fs_s = final_scores.stride()
    stride_gi_b, stride_gi_s = global_indices.stride()

    grid = (B,)

    BLOCK_S = 64
    BLOCK_D = 32

    dsa_indexer_kernel[grid](
        q, K_all, weights, seq_lens, block_table,
        final_scores, global_indices,
        B, H, D, page_size, num_pages, MNP,
        stride_q_b, stride_q_h, stride_q_d,
        stride_K_np, stride_K_p, stride_K_d,
        stride_w_b, stride_w_h,
        stride_bt_b, stride_bt_np,
        stride_fs_b, stride_fs_s,
        stride_gi_b, stride_gi_s,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    topk_indices = torch.full(
        (B, K_val), -1, device=device, dtype=torch.int32
    )
    
    for b in range(B):
        L = int(seq_lens[b].item())
        if L <= 0:
            continue

        scores_b = final_scores[b, :L]        # [L]
        global_idx_b = global_indices[b, :L]  # [L]

        actual_k = min(K_val, L)
        
        # PyTorch 的 TopK 映射
        topk_vals, topk_pos = torch.topk(scores_b, actual_k)
        topk_global = global_idx_b[topk_pos]

        topk_indices[b, :actual_k] = topk_global.to(torch.int32)

    return (topk_indices,)