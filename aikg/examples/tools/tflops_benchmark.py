import torch
import time

def benchmark(device, dtype, use_autocast, size=4096, warmup=10, iterations=10):
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    
    with torch.no_grad():
        for i in range(warmup + iterations):
            if i == warmup:
                torch.cuda.synchronize()
                start = time.time()
            if use_autocast:
                with torch.amp.autocast("cuda", enabled=True):
                    torch.matmul(a, b)
            else:
                torch.matmul(a, b)
            torch.cuda.synchronize()
            
    tflops = (2 * iterations * size**3) / (time.time() - start) / 1e12
    return tflops

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"{torch.cuda.get_device_name(device)}")

    size, warmup, iters = 8192, 10, 10

    tests = {
        # "CUDA FP64":        {"dtype": torch.float64,  "autocast": False, "allow_tf32": False},
        "CUDA FP32":        {"dtype": torch.float32,  "autocast": False, "allow_tf32": False},
        # "CUDA FP16":        {"dtype": torch.float16,  "autocast": False, "allow_tf32": False},
        # "CUDA BF16":        {"dtype": torch.bfloat16, "autocast": False, "allow_tf32": False},
        # "CUDA TF32":        {"dtype": torch.float32,  "autocast": True,  "allow_tf32": True},
        # "Mixed Precision":  {"dtype": torch.float32,  "autocast": True,  "allow_tf32": False},
    }

    print("开始性能测试...\n")
    for name, params in tests.items():
        torch.backends.cuda.matmul.allow_tf32 = params["allow_tf32"]
        try:
            tflops = benchmark(device, params["dtype"], params["autocast"], size, warmup, iters)
            print(f"{name}: {tflops:.2f} TFLOPS")
        except Exception as e:
            print(f"{name}: 测试失败 - {str(e)}")