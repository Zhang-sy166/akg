import os
import json
import random
import logging
from typing import List, Dict
from pathlib import Path

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.common_utils import get_md5_hash
# from ai_kernel_generator.utils.evolve.evolution_processors import EvolveRuntimeConfig
from ai_kernel_generator.database.island import Island

from ai_kernel_generator.database.early_stopping import NCU_METRIC_LIST, IterationRecord, EarlyStoppingDecision, BranchEarlyStoppingJudge
from ai_kernel_generator.core.worker.local_worker import ncu_json_get_mean

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
            # island_evolve_shortcut = self.evolve_from_shortcut(evolve_config.evolve_database, evolve_config.num_islands)
            island_evolve_shortcut = [[] * evolve_config.num_islands] # disable evolve from checkpoint/shortcut
            self.island_list = [Island(i, self.database_path, database_config, island_evolve_shortcut[i], evolve_config.evolve_database) for i in range(evolve_config.num_islands)]
            self.fall_back_depth = 3    # 回退深度
            
            self._initialized = True
        finally:
            self.__class__._lock = False
    
    def evolve_from_shortcut(self, evolve_database: str, num_islands: int) -> List[List[str]]:
        shortcut_path = Path(get_project_root()).parent.parent / "evolve_database" / evolve_database
        if not os.path.exists(shortcut_path):
            os.makedirs(shortcut_path, exist_ok=True)
            return [[] * num_islands]
        all_seeds = [seed for seed in os.listdir(shortcut_path) if os.path.isdir(os.path.join(shortcut_path, seed)) and 'vector_store' not in seed]
        if len(all_seeds) == 0:
            return [[] * num_islands]
        
        random.shuffle(all_seeds)
        n = len(all_seeds)
        base_size = n // num_islands
        remainder = n % num_islands
        
        result = []
        start = 0
        for i in range(num_islands):
            # 计算当前份的大小
            current_size = base_size + (1 if i < remainder else 0)
            
            if current_size > 0:
                result.append(all_seeds[start:start + current_size])
            else:
                result.append([])  # 不够分则返回空列表
            start += current_size
        
        return result
        
    def get_island(self, island_idx: int) -> Island:
        return self.island_list[island_idx]
    
    def is_island_empty(self, island_idx: int) -> bool:
        return len(self.island_list[island_idx].program_list) == 0
    
    def die_programs(self):
        # TODO
        logger.info("Make programs die of every island")
        
    def migration(self):
        # TODO
        logger.info("pd migration")
    
    def sample_island_parent(self, island_idx: int):
        # TODO
        logger.info(f"sample island_{island_idx} parent")
        
        return self.island_list[island_idx].sample_latest()
        return self.island_list[island_idx].random_sample_parent()
    
    def sample_island_others(self, island_idx: int, parent_idx: int, sample_num: int):
        # TODO
        logger.info(f"sample island_{island_idx} others")
        
        return self.island_list[island_idx].random_sample_others(parent_idx, sample_num)
    
    async def insert_island(self, island_idx: int, impl_code: str, framework_code: str, profile: str, backend: str, arch: str, dsl: str, impl_info: dict):
        # TODO
        logger.info(f"insert program into island_{island_idx}")
        await self.island_list[island_idx].insert(impl_code, framework_code, profile, backend, arch, dsl, impl_info)
    
    def judge_early_stopping(self, island_idx: int, target_id: str) -> EarlyStoppingDecision:
        # TODO
        logger.info(f"judge early stopping of island_{island_idx} with target_id {target_id}")
        
        # 1. 获取当前算子历史列表
        code_list = self.island_list[island_idx].get_branch_from_root(target_id)
        iter_record_list = []
        depth = 0
        for code in code_list:
            ncu_metric_mean = ncu_json_get_mean(json.loads(code.get_impl_info().get("ncu_profile_metric")))
            ncu_metric_mean_filter = { key: ncu_metric_mean[key] for key in ncu_metric_mean.keys() if key in NCU_METRIC_LIST }
            iter_record = IterationRecord(
                depth,
                float(code.get_impl_info().get("profile", {}).get("speedup", 0.0)),
                ncu_metric_mean_filter
            )
            iter_record_list.append(iter_record)
            depth += 1
        # 2. 调用早停判断接口，返回是否早停
        early_stopping_judger = BranchEarlyStoppingJudge()
        early_stopping_decision = early_stopping_judger.judge(code_list)
        return early_stopping_decision
        
    def fall_back_search_parent_candidate(self, island_idx: int, stop_program_id: str) -> str | None:
        """
        父代候选 stop_program_id 被判定收敛需要停止在其分支继续进化时调用该函数
        从 stop_program_id 所在分支回退，寻找可能的父代候选
        对可能的父代候选进行收敛判定，若不收敛，则返回，若仍收敛则持续回退。
        """
        fall_back_depth = self.fall_back_depth
        fall_back_candidate_list = self.island_list[island_idx].get_fall_back_candidate_list(
            stop_program_id, fall_back_depth
        )  # [program_id:str, ... ]
        # 若回退到的【父代候选】收敛，则持续回退；持续回退时，步长固定为1；
        while len(fall_back_candidate_list) != 0:
            for candidate in fall_back_candidate_list:
                if self.judge_early_stopping(island_idx, candidate).stop is False:
                    return candidate
            fall_back_depth = 1
            fall_back_candidate_list = self.island_list[island_idx].get_fall_back_candidate_list(stop_program_id, fall_back_depth)
        return None
    
    def search_parent(self, island_idx: int) -> str:
        """
        用作在某个检查点处重启进化流程时，第一轮需要搜索父代，找到一个开启点
        1. 找到所有未探索过的叶子结点，按层数降序排序，层数相同按照profile speedup值降序排序
        2. 对叶子结点列表依次判定是否收敛，返回第一个不收敛的节点
        """
        leaf_programs = []
        # 收集所有叶子节点（没有子节点的节点）
        for program in self.island_list[island_idx].program_list:
            program_id = program.get_impl_info()['id']
            # 判断是叶子结点
            if not self.island_list[island_idx].has_child(program_id):
                # 获取深度（从根到当前节点的节点数）
                depth = len(self.island_list[island_idx].get_branch_from_root(program_id))
                speedup = program.get_impl_info().get('profile', {}).get('speedup', 0.0)
                leaf_programs.append((program_id, depth, speedup))
        
        # 按深度降序，同深度按 speedup 降序排序
        leaf_programs.sort(key=lambda x: (-x[1], -x[2]))
        
        # 依次检查是否收敛，返回第一个未收敛的节点
        for program_id, _, _ in leaf_programs:
            decision = self.judge_early_stopping(island_idx, program_id)
            if not decision.stop:
                return program_id
        
        return None
    
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
            op_temp += f"第{i+1}次迭代的性能数据如下:\n" + "运行时间: " + str(op[1]["gen_time"]) + "us\n"
            op_temp += f"针对第{i+1}次迭代的优化方向建议如下，该优化方向会在下一次算子实现中被应用:\n" + op[2] + "\n"
            optimize_history_str += op_temp
        return optimize_history_str
        
        
def get_database_dir(database_dir: str='', evolve_config=None):
    if database_dir != '':
        return database_dir
    import time, os
    pd_subdir = evolve_config.op_name + '_' + str(int(time.time()))
    pd_dir = str(DEFAULT_PROGRAM_DATABASE_PATH / pd_subdir)
    os.makedirs(pd_dir, exist_ok=True)
    return pd_dir