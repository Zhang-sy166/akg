import triton
import triton.language as tl
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transposed_both_kernel(
    A_ptr,          # 指针，指向输入张量 A，形状 (K, M)
    B_ptr,          # 指针，指向输入张量 B，形状 (N, K)
    C_ptr,          # 指针，指向输出张量 C，形状 (M, N)
    M,              # 整数，A 的列数，C 的行数
    N,              # 整数，B 的行数，C 的列数
    K,              # 整数，A 的行数，B 的列数
    stride_ak,      # A 沿 K 维（第0维）的步长
    stride_am,      # A 沿 M 维（第1维）的步长
    stride_bn,      # B 沿 N 维（第0维）的步长
    stride_bk,      # B 沿 K 维（第1维）的步长
    stride_cm,      # C 沿 M 维（第0维）的步长
    stride_cn,      # C 沿 N 维（第1维）的步长
    BLOCK_SIZE_M: tl.constexpr,  # 块大小，M 维度
    BLOCK_SIZE_N: tl.constexpr,  # 块大小，N 维度
    BLOCK_SIZE_K: tl.constexpr,  # 块大小，K 维度
):
    """
    Triton 内核，计算矩阵乘法 C = A.T @ B.T，其中 A 形状 (K, M)，B 形状 (N, K)，C 形状 (M, N)。
    使用二维网格并行化，每个程序处理一个 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的输出块。
    """
    # 获取程序 ID，对应输出块的位置
    pid_m = tl.program_id(0)          # 对应 i_outer，沿 M 维度
    pid_n = tl.program_id(1)          # 对应 j_outer，沿 N 维度
    
    # 计算当前输出块的起始偏移
    i_outer = pid_m * BLOCK_SIZE_M
    j_outer = pid_n * BLOCK_SIZE_N
    
    # 初始化累加器为全零，使用 float32 精度
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 主循环：沿 K 维度分块迭代
    for k_outer in range(0, K, BLOCK_SIZE_K):
        # 创建块指针，加载 A 的转置块，形状 (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # 从 A.T 加载，即原始 A 的转置视图
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),               # A.T 的形状
            strides=(stride_am, stride_ak),  # A.T 的步长
            offsets=(i_outer, k_outer),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0)                # 行主序内存布局
        )
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))  # 形状 (BLOCK_SIZE_M, BLOCK_SIZE_K)
        
        # 创建块指针，加载 B 的转置块，形状 (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # 从 B.T 加载，即原始 B 的转置视图
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),               # B.T 的形状
            strides=(stride_bk, stride_bn),  # B.T 的步长
            offsets=(k_outer, j_outer),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0)                # 行主序内存布局
        )
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))  # 形状 (BLOCK_SIZE_K, BLOCK_SIZE_N)
        
        # 矩阵乘法累加：accumulator += a_tile @ b_tile
        accumulator = tl.dot(a_tile, b_tile, acc=accumulator)
    
    # 创建块指针，存储结果到 C 的对应块
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(i_outer, j_outer),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


class ModelNew(torch.nn.Module):
    def __init__(self):
        """初始化，无内置参数，因此为空。"""
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        执行矩阵乘法 C = A.T @ B.T，使用 Triton 内核。
        
        Args:
            A: 输入张量，形状 (K, M)，数据类型通常为 float16
            B: 输入张量，形状 (N, K)，数据类型通常为 float16
        
        Returns:
            C: 输出张量，形状 (M, N)，数据类型为 float32
        """
        # 检查形状兼容性
        K_A, M = A.shape
        N, K_B = B.shape
        assert K_A == K_B, f"矩阵维度不匹配: A 的 K={K_A}, B 的 K={K_B}"
        K = K_A
        
        # 分配输出张量 C，形状 (M, N)，使用 float32 精度，与草图一致
        C = torch.empty((M, N), dtype=torch.float32, device=A.device)
        
        # 获取输入和输出的步长（假设行主序连续布局）
        stride_ak = A.stride(0)  # A 沿 K 维的步长
        stride_am = A.stride(1)  # A 沿 M 维的步长
        stride_bn = B.stride(0)  # B 沿 N 维的步长
        stride_bk = B.stride(1)  # B 沿 K 维的步长
        stride_cm = C.stride(0)  # C 沿 M 维的步长
        stride_cn = C.stride(1)  # C 沿 N 维的步长
        
        # 定义块大小，减小以避免共享内存超限
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        # 计算网格大小：二维网格，覆盖所有输出块
        grid_m = triton.cdiv(M, BLOCK_SIZE_M)
        grid_n = triton.cdiv(N, BLOCK_SIZE_N)
        grid = (grid_m, grid_n)
        
        # 启动 Triton 内核
        matmul_transposed_both_kernel[grid](
            A, B, C,               # 输入输出张量
            M, N, K,               # 形状参数
            stride_ak, stride_am,  # A 的步长
            stride_bn, stride_bk,  # B 的步长
            stride_cm, stride_cn,  # C 的步长
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return C