from ai_kernel_generator.database.database import Database
from ai_kernel_generator.database.coder_database import CoderDatabase

from pathlib import Path

async def main():
    coder_database = CoderDatabase(config={'agent_model_config': {'feature_extractor': 'smallai_claude-sonnet-4-5-thinking-all'}})
    origin_database_path = Path('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/triton_database/nvidia/TritonBench_G_v1')
    cnt = 0
    all_py = [py for py in origin_database_path.glob('*.py')]
    all_py.sort()
    for f in all_py:
        converted = [l.strip() for l in open('converted.txt', 'r').readlines()]
        if str(f) not in converted:
            if cnt % 5 == 0 or 'flash_attn' in str(f):
                print(cnt, f)
                impl_code = ''.join(open(f, 'r').readlines())
                await coder_database.insert(impl_code, '', 'cuda', 'a100', 'triton_cuda', 'torch')
                with open('converted.txt', 'a') as converted_record:
                    converted_record.write(str(f)+'\n')
        cnt += 1
        

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())