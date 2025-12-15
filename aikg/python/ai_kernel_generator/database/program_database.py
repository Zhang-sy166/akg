import logging
from typing import List, Dict
from pathlib import Path

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash
# from ai_kernel_generator.utils.evolve.evolution_processors import EvolveRuntimeConfig
from ai_kernel_generator.database.island import Island


logger = logging.getLogger(__name__)

# Path(get_project_root()).parent.parent /mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/
DEFAULT_PROGRAM_DATABASE_PATH = Path(get_project_root()).parent.parent / "program_database"


class ProgramDatabase():
    _instance: Dict[str, 'ProgramDatabase'] = {}
    _lock = False   # True == occupied; False == free
    
    def __new__(cls, database_path: str='', evolve_config=None):
        database_path = get_database_dir(database_path, evolve_config)
        instance_key = get_md5_hash(database_path=database_path)
        
        if instance_key not in cls._instance or cls._instance[instance_key] is None:
            while cls._lock:
                pass
            cls._lock = True
            try:
                if instance_key not in cls._instance or cls._instance[instance_key] is None:
                    cls._instance[instance_key] = object.__new__(cls)
            finally:
                cls._lock = False
        return cls._instance[instance_key]
    
    def __init__(self, database_path: str='', evolve_config=None):
        while self.__class__._lock:
            pass
        self.__class__._lock = True
        
        try:
            if hasattr(self, '_initialized') and self._initialized:
                return
            
            self.database_path = get_database_dir(database_path, evolve_config)
            database_config = evolve_config.config
            self.island_list = [Island(i, self.database_path, database_config) for i in range(evolve_config.num_islands)]
            
            self._initialized = True
        finally:
            self.__class__._lock = False
    
    def die_programs(self):
        # TODO
        logger.info("Make programs die of every island")
        
    def migration(self):
        # TODO
        logger.info("pd migration")
    
    def sample_island_parent(self, island_idx: int):
        # TODO
        logger.info(f"sample island_{island_idx} parent")
        
        return self.island_list[island_idx].random_sample_parent()
    
    def sample_island_others(self, island_idx: int, parent_idx: int, sample_num: int):
        # TODO
        logger.info(f"sample island_{island_idx} others")
        
        return self.island_list[island_idx].random_sample_others(parent_idx, sample_num)
    
    async def insert_island(self, island_idx: int, impl_code: str, framework_code: str, profile: str, backend: str, arch: str, dsl: str, impl_info: dict):
        # TODO
        logger.info(f"insert program into island_{island_idx}")
        await self.island_list[island_idx].insert(impl_code, framework_code, profile, backend, arch, dsl, impl_info)
    
    def build_optimize_history(self, parent_id: str):
        optimize_history = [] # [ [impl_code, speed_profile, porfiler_suggestion] ... ]
        while parent_id:
            for island in self.island_list:
                program = island.find_program_by_id(parent_id)
                if program:
                    info = program.get_impl_info()
                    optimize_history.insert(
                        0,
                        [info.get("impl_code", ""), info.get("profile", {}), info.get("ncu_profile_result", "")]
                    )
                    parent_id = info.get("parent_id", None)
                    break
        optimize_history_str = "下面是该算子的历史优化路径（包括相对应的算子实现、性能测试和优化建议），根据时间顺序从开始到现在排列：\n\n"
        for i, op in enumerate(optimize_history):
            op_temp = f"第{i+1}次迭代的算子实现如下:\n" + op[0] + "\n"
            op_temp += f"第{i+1}次迭代的性能数据如下:\n" + "运行时间: " + op[1]["gen_time"] + "us\n"
            op_temp += f"针对第{i+1}次迭代的优化方向建议如下，该优化方向会在下一次算子实现中被应用:\n" + op[2] + "\n"
            optimize_history_str += op_temp
        
        return optimize_history
        
        
def get_database_dir(database_dir: str='', evolve_config=None):
    if database_dir != '':
        return database_dir
    import time, os
    pd_subdir = evolve_config.op_name + '_' + str(int(time.time()))
    pd_dir = str(DEFAULT_PROGRAM_DATABASE_PATH / pd_subdir)
    os.makedirs(pd_dir, exist_ok=True)
    return pd_dir