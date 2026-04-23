from pathlib import Path
import os
import subprocess
import sys

level = 'level2'
kernel_no = [i for i in range(1, 100, 8)]
# import pdb;pdb.set_trace()
base_dir = Path('/mnt/lustre-client/zhangzizheng/AIKG/KernelBench/KernelBench')

for kno in kernel_no:
    # import pdb;pdb.set_trace()
    op_name = [d for d in os.listdir(base_dir / level) if d.startswith(str(kno)+'_')][0]
    
    task_desc = base_dir/level/op_name
    evolve_database = level + '/' + op_name.replace('.py', '')
    op_name_without_no = ('_'.join(op_name.split('_')[1:])).replace('.py', '')
    print('\n')
    print('op_name:\t\t', op_name_without_no)
    print('task_desc:\t\t', task_desc)
    print('evolve_database:\t', evolve_database)
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['AIKG_SMALLAI_API_KEY'] = 'sk-gDECyQGeilKJQplndfXM2rXviL748hNhL9Yn57lKAV9wetQr'
    python_exe = sys.executable  
    cmd = [
        python_exe, "run_torch_evolve_triton.py",
        "--op-name", op_name_without_no,
        "--task-desc", task_desc,
        "--evolve-database", evolve_database
    ]
    stdout_log = 'log/' + level + '/' + str(kno) + '_log.txt'
    stderr_log = 'log/' + level + '/' + str(kno) + '_err.txt'
    # log_dir = os.path.dirname(stdout_log)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #     print(f"创建日志目录: {log_dir}")
    # err_dir = os.path.dirname(stderr_log)
    # if not os.path.exists(err_dir):
    #     os.makedirs(err_dir)
    #     print(f"创建日志目录: {err_dir}")
    with open(stdout_log, "w", encoding="utf-8") as f_out, open(stderr_log, "w", encoding="utf-8") as f_err:
        result = subprocess.run(
                cmd,
                env=env,
                stdout=f_out,
                stderr=f_err,
                shell=False,  # 若cmd是列表（如["ls", "-l"]），建议设为False
                check=True,
                text=True,    # 以文本模式处理输出（替代universal_newlines=True）
            )
        
        print(f"{kno}命令执行完成！退出码: {result.returncode}")
        print(f"标准输出已写入: {stdout_log}")
        print(f"错误输出已写入: {stderr_log}")    
    
    
