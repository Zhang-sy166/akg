import json
from safetensors.torch import load_file
import torch
from pathlib import Path
import requests


def load_safetensor_from_url(url: str, tensor_file: str, tensor_name: str, local_cache_dir: str = "dsa_inputs") -> torch.Tensor:
    """
    从URL加载safetensor文件（优先使用本地缓存，不存在则下载）
    
    Args:
        url: 远程文件URL
        local_cache_dir: 本地缓存根目录
    
    Returns:
        加载后的torch tensor
    """    
    # 构建本地缓存路径
    cache_path = Path(local_cache_dir) / tensor_file
    
    # 检查文件是否存在
    if not cache_path.exists():
        # 文件不存在则下载
        # 开始下载
        try:
            print(f"开始下载文件: {url} -> {cache_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 抛出HTTP错误
            
            # 分块写入文件（避免大文件占用过多内存）
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"文件下载完成: {cache_path}")
        
        except Exception as e:
            raise RuntimeError(f"下载文件失败 {url}: {str(e)}")
    
    # 加载safetensor文件
    try:
        tensor_dict = load_file(str(cache_path))
        first_tensor_key = tensor_dict[tensor_name]
        return first_tensor_key
    except Exception as e:
        raise RuntimeError(f"加载safetensor文件失败 {cache_path}: {str(e)}")

def get_dsa_input():
    input_list = open('/mnt/lustre-client/zhangzizheng/ALL_DATASETS/flashinfer/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl', 'r').readlines()
    input_ret_list = []
    torch.manual_seed(0)
    for input in input_list:
        device = 'cuda'
        input_json = json.loads(input)
        
        # static value 
        num_index_heads = 64
        index_head_dim = 128
        page_size = 64
        
        # var value
        batch_size = input_json['workload']['axes']['batch_size']
        max_num_pages = input_json['workload']['axes']['max_num_pages']
        num_pages = input_json['workload']['axes']['num_pages']
        
        # load from tensor
        seq_lens_url = 'https://hf-mirror.com/datasets/flashinfer-ai/flashinfer-trace/resolve/main/' + input_json['workload']['inputs']['seq_lens']['path'] + '?download=true'
        block_table_url = 'https://hf-mirror.com/datasets/flashinfer-ai/flashinfer-trace/resolve/main/' + input_json['workload']['inputs']['block_table']['path'] + '?download=true'
        seq_lens = load_safetensor_from_url(seq_lens_url, input_json['workload']['inputs']['seq_lens']['path'].split('/')[-1], 'seq_lens')
        block_table = load_safetensor_from_url(block_table_url, input_json['workload']['inputs']['block_table']['path'].split('/')[-1], 'block_table')
        seq_lens = seq_lens.to(device)
        block_table = block_table.to(device)
        
        # radom tensor
        q_index_fp8 = torch.randn(batch_size, num_index_heads, index_head_dim, 
                                device=device, dtype=torch.float32).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        
        k_fp8 = torch.randn(num_pages, page_size, index_head_dim,
                                        device=device, dtype=torch.float32).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_scale = torch.randn(num_pages, page_size, 1, device=device, dtype=torch.float32) * 0.1
        k_fp8_bytes = k_fp8.view(torch.int8).view(num_pages, -1) # [num_pages, page_size * 1 * 128]
        k_scale_bytes = k_scale.view(torch.int8).view(num_pages, -1) # [num_pages, page_size * 4]
        k_index_cache_fp8 = torch.cat([k_fp8_bytes, k_scale_bytes], dim=-1).view(num_pages, page_size, 1, -1) # [num_pages, page_size, 1, head_dim_with_scale]
        
        weights = torch.randn(batch_size, num_index_heads, device=device, dtype=torch.float32)
        
        input_ret_list.append((q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table))
    return input_ret_list

def get_dsa_input_new():
    input_list = open('/mnt/lustre-client/zhangzizheng/AIKG/flash_infer_datasets/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl', 'r').readlines()
    input_ret_list = []
    torch.manual_seed(0)
    for input in input_list:
        device = 'cuda'
        input_json = json.loads(input)
        
        # static value 
        num_index_heads = 64
        index_head_dim = 128
        page_size = 64
        
        # var value
        batch_size = input_json['workload']['axes']['batch_size']
        max_num_pages = input_json['workload']['axes']['max_num_pages']
        num_pages = input_json['workload']['axes']['num_pages']
        
        # load from tensor
        seq_lens_url = None
        block_table_url = None
        # import pdb;pdb.set_trace()
        seq_lens = load_safetensor_from_url(seq_lens_url, input_json['workload']['inputs']['seq_lens']['path'].split('/')[-1], 'seq_lens', '/mnt/lustre-client/zhangzizheng/AIKG/flash_infer_datasets/blob/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64')
        block_table = load_safetensor_from_url(block_table_url, input_json['workload']['inputs']['block_table']['path'].split('/')[-1], 'block_table', '/mnt/lustre-client/zhangzizheng/AIKG/flash_infer_datasets/blob/workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64')
        seq_lens = seq_lens.to(device)
        block_table = block_table.to(device)
        
        # radom tensor
        q_index_fp8 = torch.randn(batch_size, num_index_heads, index_head_dim, 
                                device=device, dtype=torch.float32).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        
        k_fp8 = torch.randn(num_pages, page_size, index_head_dim,
                                        device=device, dtype=torch.float32).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_scale = torch.randn(num_pages, page_size, 1, device=device, dtype=torch.float32) * 0.1
        k_fp8_bytes = k_fp8.view(torch.int8).view(num_pages, -1) # [num_pages, page_size * 1 * 128]
        k_scale_bytes = k_scale.view(torch.int8).view(num_pages, -1) # [num_pages, page_size * 4]
        k_index_cache_fp8 = torch.cat([k_fp8_bytes, k_scale_bytes], dim=-1).view(num_pages, page_size, 1, -1) # [num_pages, page_size, 1, head_dim_with_scale]
        
        weights = torch.randn(batch_size, num_index_heads, device=device, dtype=torch.float32)
        
        input_ret_list.append((q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table))
        
        q_index_fp8_new = torch.load("q_index_fp8.pt")
        k_index_cache_fp8_new = torch.load("k_index_cache_fp8.pt")
        weights_new = torch.load("weights.pt")
        seq_lens_new = torch.load("seq_lens.pt")
        block_table_new = torch.load("block_table.pt")
        # import pdb;pdb.set_trace()
        input_ret_list.append((q_index_fp8_new, k_index_cache_fp8_new, weights_new, seq_lens_new, block_table_new))
        break
    return input_ret_list

if __name__ == '__main__':
    out = get_dsa_input_new()
    import pdb;pdb.set_trace()
    