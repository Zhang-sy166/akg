import torch
import time
import subprocess
import os

cnt=0
while True:
    if cnt % 100 == 0:
        print(f"I'm waiting for GPU_{os.environ.get('CUDA_VISIBLE_DEVICES', 0)} resources for {cnt * 300 / 60} minutes ...\n")
        result = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id=1",
                "--query-compute-apps=pid"
            ],
            encoding="utf-8",
            stderr=subprocess.STDOUT
        )
        if len(result.split()) == 1:
            break
        time.sleep(3)
print(result)