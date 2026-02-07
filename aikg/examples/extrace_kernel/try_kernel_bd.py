import ast
import inspect
from typing import Dict, List, Tuple

class TritonKernelExtractor(ast.NodeVisitor):
    """
    Triton算子提取器，基于AST解析提取三部分内容：
    1. 核心Triton计算片段（带@triton.jit装饰器的函数）
    2. 数据预处理与启动维度设置部分（调用kernel前的逻辑）
    3. 外层暴露函数（封装调用逻辑的顶层函数）
    """
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)
        
        # 存储提取结果
        self.triton_kernels: Dict[str, str] = {}  # 核心triton计算片段
        self.preprocessing_code: Dict[str, str] = {}  # 预处理+启动维度设置
        self.outer_functions: Dict[str, str] = {}  # 外层暴露函数
        
        # 辅助变量
        self.lines = source_code.splitlines()
        self.triton_jit_decorators = ('@triton.jit', '@triton.compiler.jit')

    def get_code_segment(self, node: ast.AST) -> str:
        """根据AST节点的起止位置提取代码片段"""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            # 注意：lineno是从1开始的，切片是左闭右开
            return '\n'.join(self.lines[node.lineno-1:node.end_lineno]).strip()
        return ""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """遍历函数定义节点，区分不同类型的函数"""
        # 1. 判断是否是核心Triton Kernel（带@triton.jit装饰器）
        is_triton_kernel = False
        for decorator in node.decorator_list:
            decorator_code = self.get_code_segment(decorator)
            if any(jit_decorator in decorator_code for jit_decorator in self.triton_jit_decorators):
                is_triton_kernel = True
                break
        
        if is_triton_kernel:
            # 提取核心Triton计算片段
            kernel_code = self.get_code_segment(node)
            self.triton_kernels[node.name] = kernel_code
        else:
            # 2. 处理外层暴露函数
            outer_func_code = self.get_code_segment(node)
            self.outer_functions[node.name] = outer_func_code
            
            # 3. 提取函数内的预处理和启动维度设置逻辑
            # 查找调用triton kernel的代码行，分离预处理部分
            preprocessing_lines = []
            in_preprocessing = True
            
            for stmt in node.body:
                stmt_code = self.get_code_segment(stmt)
                # 判断是否是调用triton kernel的语句（特征：包含kernel启动逻辑，如grid、num_warps等）
                is_kernel_launch = any(keyword in stmt_code for keyword in ['grid', 'num_warps', 'num_stages', '()'])
                
                if in_preprocessing and not is_kernel_launch:
                    preprocessing_lines.append(stmt_code)
                elif is_kernel_launch:
                    # 启动维度设置也归为预处理部分
                    preprocessing_lines.append(stmt_code)
                    in_preprocessing = False  # 后续是kernel调用，不再收集
            
            self.preprocessing_code[node.name] = '\n'.join(preprocessing_lines).strip()
        
        # 继续遍历子节点
        self.generic_visit(node)

    def extract_all(self) -> Dict[str, Dict[str, str]]:
        """执行提取并返回所有结果"""
        self.visit(self.tree)
        return {
            "核心Triton计算片段": self.triton_kernels,
            "数据预处理与启动维度设置": self.preprocessing_code,
            "外层暴露函数": self.outer_functions
        }

def extract_triton_segments(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    从Triton算子文件中提取三部分内容的主函数
    
    参数:
        file_path: Triton算子文件的路径
    
    返回:
        包含三部分内容的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        extractor = TritonKernelExtractor(source_code)
        results = extractor.extract_all()
        
        # 打印提取结果（可根据需要修改输出格式）
        for part_name, content in results.items():
            print(f"\n========== {part_name} ==========")
            for func_name, code in content.items():
                print(f"\n【{func_name}】:")
                print(code)
        import pdb;pdb.set_trace()
        return results
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return {}
    except SyntaxError as e:
        print(f"错误：文件语法错误 - {e}")
        return {}

# 示例使用
if __name__ == "__main__":
    # 替换为你的Triton算子文件路径
    triton_file_path = "/mnt/lustre-client/zhangzizheng/AIKG/TritonBench/data/TritonBench_G_v1/parallel_attention.py"
    extract_triton_segments(triton_file_path)