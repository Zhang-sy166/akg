import torch
import torch.nn as nn
import triton
import triton.language as tl


class Model(nn.Module):
    """
    Triton-based DSA TopK indexer with FP8 quantization for DeepSeek-V3.2.
    约束：
      - num_index_heads = 64
      - index_head_dim  = 128
      - page_size       = 64
      - k_index_cache_fp8 形状: [num_pages, page_size, 1, 132] (FP8 128 字节 + scale 4 字节)
      - q_index_fp8 形状: [batch, 64, 128] (torch.float8_e4m3fn)
    """

    def __init__(self, topk: int = 2048, page_size: int = 64):
        super().__init__()
        self.topk = topk
        self.page_size = page_size

    def dequant_fp8_kv_cache(self, k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
        """保持原来的 PyTorch 实现。"""
        k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
        num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4  # 128
        # import pdb;pdb.set_trace()
        kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

        fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
        fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
        fp8_float = fp8_tensor.to(torch.float32)

        scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
        scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]

        return fp8_float * scale  # [num_pages, page_size, 128], float32
    
    def forward(self, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
        """
        Native Sparse Attention (DSA) TopK indexer with FP8 quantization for DeepSeek-V3.2. 
        Computes sparse attention scores using ReLU activation and learned weights, then selects top-K KV cache indices. 
        Formula: sum(relu(q @ K.T) * weights). Matches SGLang/deep_gemm implementation. Page size 64 variant.

        Returns:
            (torch.Tensor,): topk_indices with shape [batch_size, topk] and dtype int32.
        """
        batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
        num_pages, page_size, _, _ = k_index_cache_fp8.shape
        topk = 2048

        # Check constants
        assert num_index_heads == 64
        assert index_head_dim == 128
        assert page_size == 64

        device = q_index_fp8.device

        # Dequantize inputs
        q = q_index_fp8.to(torch.float32)  # [batch, heads, head_dim]
        K_all = self.dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, head_dim]
        # import pdb;pdb.set_trace()

        topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
        max_num_pages = block_table.shape[1]
        final_scores_tensors = []
        for b in range(batch_size):
            seq_len = int(seq_lens[b].item())
            
            if seq_len == 0:
                continue

            # Get pages for this sequence
            num_pages_for_seq = (seq_len + page_size - 1) // page_size
            page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
            
            # Gather K from pages
            K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
            K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, head_dim]
            
            # Query for this batch element
            q_b = q[b]  # [num_heads, head_dim]
            
            # Compute attention scores
            scores = q_b @ K.T  # [num_heads, seq_len]
            
            # Apply ReLU (deep_gemm uses ReLU activation)
            scores_relu = torch.relu(scores)  # [num_heads, seq_len]
            
            # Apply learned weights and sum across heads
            w = weights[b]  # [num_heads]
            weighted_scores = scores_relu * w[:, None]  # [num_heads, seq_len]
            # import pdb;pdb.set_trace()
            final_scores = weighted_scores.sum(dim=0)  # [seq_len]
            
            # Select top-K
            actual_topk = min(topk, seq_len)
            _, topk_idx = torch.topk(final_scores, actual_topk)
            
            # Convert to global token indices
            # Token index = page_idx * page_size + offset_in_page
            page_idx_per_token = topk_idx // page_size
            offset_per_token = topk_idx % page_size
            global_page_idx = page_indices[page_idx_per_token]
            topk_tokens = global_page_idx * page_size + offset_per_token
            
            topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
            final_scores_tensors.append(final_scores)

        # return (topk_indices, final_scores_tensors)
        return (topk_indices,)

# ========================= Triton Kernel =========================

@triton.jit
def dsa_indexer_kernel(
    # 输入参数
    q_ptr,               # 查询指针 [num_heads, head_dim] float32
    K_ptr,               # 键指针 [seq_len, head_dim] float32
    scores_ptr,          # 输出分数指针 [num_heads, seq_len] float32
    num_heads,           # 头数量
    seq_len,             # 序列长度
    head_dim: tl.constexpr,  # 头维度
    BLOCK_SIZE: tl.constexpr,  # 线程块大小
):
    # 1. 获取程序ID和计算偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = num_heads * seq_len
    mask = offsets < total_elements
    
    # 2. 计算输出索引：i为头索引，j为序列位置索引
    element_idx = offsets
    i = element_idx // seq_len
    j = element_idx % seq_len
    
    # 3. 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # 4. 循环计算点积：每个线程循环head_dim次，累加q[i,k]*K[j,k]
    for k in range(0, head_dim):
        # 加载q[i, k]
        q_offset = i * head_dim + k
        q_val = tl.load(q_ptr + q_offset, mask=mask, other=0.0)
        
        # 加载K[j, k]
        K_offset = j * head_dim + k
        K_val = tl.load(K_ptr + K_offset, mask=mask, other=0.0)
        
        # 累加点积
        accumulator += q_val * K_val
    
    # 5. 存储分数
    tl.store(scores_ptr + element_idx, accumulator, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个无参数算子，因此 __init__ 为空
    
    def forward(self, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
        """
        输入:
            q_index_fp8: [batch_size, num_index_heads, index_head_dim] float8_e4m3fn (作为 torch.tensor 传入)
            k_index_cache_fp8: [num_pages, page_size, 1, head_dim_sf] int8 (解释为 uint8)
            weights: [batch_size, num_index_heads] float32
            seq_lens: [batch_size] int32
            block_table: [batch_size, max_num_pages] int32
        输出:
            topk_indices: [batch_size, topk] int32
        """
        # 从输入张量获取形状参数
        batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
        num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
        max_num_pages = block_table.shape[1]
        head_dim = head_dim_sf - 4  # 128
        topk = 2048
        
        # 检查常量，与 torch 实现一致
        assert num_index_heads == 64
        assert index_head_dim == 128
        assert page_size == 64
        assert topk == 2048
        
        device = q_index_fp8.device
        
        # 步骤1: 在 host 端反量化 FP8 KV 缓存，使用 torch 操作（匹配原始实现）
        # 注意：避免在 Triton 内核中处理复杂的 FP8 转换
        k_index_cache_fp8_uint8 = k_index_cache_fp8.view(torch.uint8)
        kv_flat = k_index_cache_fp8_uint8.view(num_pages, page_size * head_dim_sf)
        # FP8 部分: 前 page_size * head_dim 字节
        fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
        fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
        fp8_float = fp8_tensor.to(torch.float32)  # [num_pages, page_size, head_dim]
        # Scale 部分: 后 page_size * 4 字节
        scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
        scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)  # [num_pages, page_size, 1]
        K_all = fp8_float * scale  # [num_pages, page_size, head_dim]
        
        # 步骤2: 将查询转换为 float32
        q = q_index_fp8.to(torch.float32)  # [batch_size, 64, 128]
        
        # 步骤3: 为输出分配内存
        topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
        
        # 步骤4: 循环处理每个批次（由于序列长度动态变化）
        for b in range(batch_size):
            seq_len = int(seq_lens[b].item())
            if seq_len == 0:
                continue
            
            # 获取当前序列的页面索引
            num_pages_for_seq = (seq_len + page_size - 1) // page_size
            page_indices = block_table[b, :num_pages_for_seq].to(torch.long)  # [num_pages_for_seq]
            
            # 从 K_all 收集页面
            K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
            K = K_paged.reshape(-1, head_dim)[:seq_len]  # [seq_len, head_dim]
            
            # 当前批次的查询
            q_b = q[b]  # [64, head_dim]
            
            # 为分数矩阵分配内存
            scores = torch.empty((num_index_heads, seq_len), dtype=torch.float32, device=device)
            
            # 调用 Triton 内核计算注意力分数 q_b @ K.T
            # 总分数元素数 = num_heads * seq_len
            total_score_elements = num_index_heads * seq_len
            BLOCK_SIZE = 256  # 2的幂，根据性能调整
            grid = lambda meta: (triton.cdiv(total_score_elements, meta['BLOCK_SIZE']),)
            dsa_indexer_kernel[grid](
                q_b, K, scores,
                num_index_heads, seq_len, head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # 应用 ReLU 和加权求和（在 host 端用 torch 操作）
            scores_relu = torch.relu(scores)  # [64, seq_len]
            w = weights[b]  # [64]
            weighted_scores = scores_relu * w[:, None]  # [64, seq_len]
            final_scores = weighted_scores.sum(dim=0)  # [seq_len]
            
            # 选择 Top-K
            actual_topk = min(topk, seq_len)
            _, topk_idx = torch.topk(final_scores, actual_topk)  # 本地索引
            
            # 转换为全局令牌索引: page_idx * page_size + offset
            page_idx_per_token = topk_idx // page_size
            offset_per_token = topk_idx % page_size
            global_page_idx = page_indices[page_idx_per_token]  # 使用页面索引映射
            topk_tokens = global_page_idx * page_size + offset_per_token
            
            topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
        
        return (topk_indices,)

if __name__ == "__main__":
    with torch.no_grad():
        # q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = get_inputs()
        from get_dsa_input import get_dsa_input_new
        inputs = get_dsa_input_new()

        torch_model = Model()
        # from baseline import run as stand_run
        # stand_model = stand_run
        # triton_model = ModelNew()
        from kernel import run
        triton_model = run
        cnt_error = 0
        cnt_total = 0
        for i, input in enumerate(inputs):
            # i == 19 b = 2 973 974 wrong
            # if i == 49:
            #     import pdb;pdb.set_trace()
            # out_ref, final_scores_ref = torch_model(*input)
            # out_tri, final_scores_tri = triton_model(*input)
            out_ref,  = torch_model(*input)
            # out_ref,  = stand_model(*input)
            out_tri,  = triton_model(*input)
            # import pdb;pdb.set_trace()
            # print(input[3])
            # print(f"{i} max diff:", (out_ref.to(torch.int64) - out_tri.to(torch.int64)).abs().max())
            # print(f"{i} out allclose:", torch.equal(out_ref, out_tri))
            
            if torch.equal(out_ref, out_tri) == False:
                # print(f"----- wrong ----- {i} max diff:", max([(ref - tri).abs().max() for ref, tri in zip(final_scores_ref, final_scores_tri)]))
                print(f"----- wrong {i} -----")
                # import pdb;pdb.set_trace()
                # print(torch.allclose(torch.cat(final_scores_torch), torch.cat(final_scores_triton)))
                cnt_error += 1
                # if i == 119:
                #     import pdb;pdb.set_trace()
            else:
                # import pdb;pdb.set_trace()
                # print(f"{i} max diff:", max([(ref - tri).abs().max() for ref, tri in zip(final_scores_ref, final_scores_tri)]))
                pass
            cnt_total += 1
        print(f"Total: {cnt_total}, Error: {cnt_error}, Success Rate: {(cnt_total-cnt_error)/cnt_total:.2%}")