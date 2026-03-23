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

        return (topk_indices, )

# ========================= Triton Kernel =========================

@triton.jit
def dsa_indexer_kernel(
    q_ptr, 
    K_T_ptr,  # [head_dim, seq_len]
    weights_ptr, 
    scores_ptr, 
    seq_len, 
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    计算每个token的加权注意力分数。
    
    输入:
        q_ptr: [num_heads, head_dim] float32
        K_T_ptr: [head_dim, seq_len] float32 (K的转置)
        weights_ptr: [num_heads] float32
        scores_ptr: [seq_len] float32 (输出分数)
        seq_len: int32
    """
    pid_m = tl.program_id(0)  # head block index
    pid_n = tl.program_id(1)  # token block index
    
    # 计算当前block处理的head范围和token范围
    q_head_start = pid_m * BLOCK_SIZE_M
    token_start = pid_n * BLOCK_SIZE_N
    
    # 创建边界掩码
    q_head_mask = q_head_start + tl.arange(0, BLOCK_SIZE_M) < num_heads
    token_mask = token_start + tl.arange(0, BLOCK_SIZE_N) < seq_len
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿head_dim维度循环，使用子块大小32
    for k in range(0, head_dim, 32):
        # 加载q的块: [BLOCK_SIZE_M, 32]
        q_offsets = (q_head_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) * head_dim + \
                    k + tl.arange(0, 32)[None, :]
        q_mask = q_head_mask[:, None] & (tl.arange(0, 32)[None, :] < head_dim - k)
        q_block = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        
        # 加载K_T的块: [32, BLOCK_SIZE_N]
        K_offsets = (k + tl.arange(0, 32)[:, None]) * seq_len + \
                    (token_start + tl.arange(0, BLOCK_SIZE_N)[None, :])
        K_mask = (tl.arange(0, 32)[:, None] < head_dim - k) & token_mask[None, :]
        K_block = tl.load(K_T_ptr + K_offsets, mask=K_mask, other=0.0)
        
        # 矩阵乘法累加
        accumulator += tl.dot(q_block, K_block)
    
    # 应用ReLU激活
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # 加载权重并加权
    weights = tl.load(weights_ptr + q_head_start + tl.arange(0, BLOCK_SIZE_M), 
                     mask=q_head_mask, other=0.0)
    weighted_accumulator = accumulator * weights[:, None]
    
    # 归约求和: 跨head维度求和，得到每个token的分数
    block_sum = tl.sum(weighted_accumulator, axis=0)  # [BLOCK_SIZE_N]
    
    # 原子操作写回全局内存中的分数数组
    output_offsets = token_start + tl.arange(0, BLOCK_SIZE_N)
    tl.atomic_add(scores_ptr + output_offsets, block_sum, mask=token_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def dequant_fp8_kv_cache(self, k_index_cache_fp8):
        """反量化FP8 KV cache，与原始实现相同"""
        k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
        num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
        head_dim = head_dim_sf - 4  # 128
        
        kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
        
        fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
        fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
        fp8_float = fp8_tensor.to(torch.float32)
        
        scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
        scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
        
        return fp8_float * scale
    
    def forward(self, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
        """
        参数:
            q_index_fp8: [batch_size, num_index_heads, index_head_dim] float8_e4m3fn
            k_index_cache_fp8: [num_pages, page_size, 1, head_dim_with_scale] int8 (interpreted as uint8)
            weights: [batch_size, num_index_heads] float32
            seq_lens: [batch_size] int32
            block_table: [batch_size, max_num_pages] int32
        返回:
            topk_indices: [batch_size, topk] int32
        """
        batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
        num_pages, page_size, _, _ = k_index_cache_fp8.shape
        topk = 2048  # 硬编码，根据任务描述
        max_num_pages = block_table.shape[1]
        device = q_index_fp8.device
        
        # 反量化输入
        q = q_index_fp8.to(torch.float32)  # [batch_size, num_index_heads, index_head_dim]
        K_all = self.dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, index_head_dim]
        
        topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
        
        # 处理每个batch
        for b in range(batch_size):
            seq_len = int(seq_lens[b].item())
            if seq_len == 0:
                continue
            
            # 获取当前序列的page索引
            num_pages_for_seq = (seq_len + page_size - 1) // page_size
            page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
            
            # 收集K矩阵并重塑
            K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, index_head_dim]
            K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, index_head_dim]
            
            # 转置K为[head_dim, seq_len]以便kernel高效访问
            K_T = K.t().contiguous()  # [index_head_dim, seq_len]
            
            # 为当前batch准备输出分数张量
            scores = torch.zeros(seq_len, dtype=torch.float32, device=device)
            
            # 获取当前batch的q和weights
            q_b = q[b].contiguous()  # [num_index_heads, index_head_dim]
            weights_b = weights[b].contiguous()  # [num_index_heads]
            
            # 启动内核计算注意力分数
            BLOCK_SIZE_M = 16  # 每个block处理的head数量
            BLOCK_SIZE_N = 64  # 每个block处理的token数量
            
            # 计算网格大小
            grid_m = triton.cdiv(num_index_heads, BLOCK_SIZE_M)
            grid_n = triton.cdiv(seq_len, BLOCK_SIZE_N)
            grid = (grid_m, grid_n)
            
            dsa_indexer_kernel[grid](
                q_b, 
                K_T, 
                weights_b, 
                scores, 
                seq_len, 
                head_dim=index_head_dim,
                num_heads=num_index_heads,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
            
            # 选择top-k token索引
            actual_topk = min(topk, seq_len)
            _, topk_idx = torch.topk(scores, actual_topk)
            
            # 转换为全局token索引: page_idx * page_size + offset_in_page
            page_idx_per_token = topk_idx // page_size
            offset_per_token = topk_idx % page_size
            global_page_idx = page_indices[page_idx_per_token]
            topk_tokens = global_page_idx * page_size + offset_per_token
            
            topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)
        
        return (topk_indices,)
if __name__ == "__main__":
    with torch.no_grad():
        # q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = get_inputs()
        from get_dsa_input import get_dsa_input
        inputs = get_dsa_input()

        torch_model = Model()
        triton_model = ModelNew()
        for i, input in enumerate(inputs):
            # i == 19 b = 2 973 974 wrong
            # if i == 49:
            #     import pdb;pdb.set_trace()
            # out_ref, final_scores_ref = torch_model(*input)
            # out_tri, final_scores_tri = triton_model(*input)
            out_ref,  = torch_model(*input)
            out_tri,  = triton_model(*input)
            # import pdb;pdb.set_trace()
            # print(input[3])
            # print(f"{i} max diff:", (out_ref.to(torch.int64) - out_tri.to(torch.int64)).abs().max())
            # print(f"{i} out allclose:", torch.equal(out_ref, out_tri))
            
            if torch.equal(out_ref, out_tri) == False:
                # print(f"----- wrong ----- {i} max diff:", max([(ref - tri).abs().max() for ref, tri in zip(final_scores_ref, final_scores_tri)]))
                print(f"----- wrong {i} -----")
                # if i == 119:
                #     import pdb;pdb.set_trace()
            else:
                # print(f"{i} max diff:", max([(ref - tri).abs().max() for ref, tri in zip(final_scores_ref, final_scores_tri)]))
                pass