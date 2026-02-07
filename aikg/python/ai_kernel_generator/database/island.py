import os
import json
import logging
import random
from pathlib import Path

from ai_kernel_generator.utils.common_utils import get_md5_hash

from ai_kernel_generator.database.database import Database
from ai_kernel_generator.database.program import Program
from ai_kernel_generator.database.evolve_database import EvolveVectorStore

logger = logging.getLogger(__name__)

class Island(Database):
    def __init__(self, island_id: int, pd_path: str, database_config: dict, evolve_shortcut: list[str], evolve_database: str):
        self.evolve_database_suffix = evolve_database
        self.island_database_path = pd_path + '/island_' + str(island_id)
        os.makedirs(self.island_database_path, exist_ok=True)
        
        # maintain an online list
        self.program_list: list[Program] = []
        
        self.move_evolve_shortcut(evolve_shortcut, evolve_database)
        
        self.basic_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='basic_vector_store',
            features=['basic'],
            config=database_config
        )
        self.schedule_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='schedule_vector_store',
            features=['schedule'],
            config=database_config
        )
        self.memory_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='memory_vector_store',
            features=['memory'],
            config=database_config
        )
        self.vector_stores = [self.basic_vector_store, self.schedule_vector_store, self.memory_vector_store]
        self.vector_store_map = {
            self.basic_vector_store: 'basic',
            self.schedule_vector_store: 'schedule',
            self.memory_vector_store: 'memory',
        }
                
        super().__init__(self.island_database_path, self.vector_stores, database_config)
        
        logger.info(f'Island {island_id} was created, has {len(evolve_shortcut)} evolve shortcuts.\n')
    
    def move_evolve_shortcut(self, evolve_shortcut: list[str], evolve_database: str):
        for es in evolve_shortcut:
            src_dir = Path(self.island_database_path).parent.parent.parent / "evolve_database" / evolve_database / es
            if os.path.exists(src_dir) and os.path.isdir(src_dir):
                des_dir = Path(self.island_database_path) / es
                os.system(f"cp -r {src_dir} {des_dir}")      
                # maintain an online list
                self.program_list.append(Program(str(des_dir / "impl_info.json")))    
    
    def sample_latest(self):
        if len(self.program_list) == 0:
            return None
        return self.program_list[-1].get_impl_info()
    
    def random_sample_parent(self):
        if len(self.program_list) == 0:
            return None
        self.program_list.sort(key=lambda x: int(x.get_impl_info()["profile"]["gen_time"]))
        return random.choice(self.program_list).get_impl_info()
    
    def random_sample_others(self, parent_id: int, sample_num: int):
        if len(self.program_list) <= 1:
            return []
        if len(self.program_list) < 1 + sample_num:
            sample_num = len(self.program_list) - 1
        program_exclude_parent_list = [p for p in self.program_list if p.get_impl_info()['id'] != parent_id]
        return [ p.get_impl_info() for p in random.choices(program_exclude_parent_list, k=sample_num)]

    
    async def insert(self, impl_code: str, framework_code: str, profile: str, backend: str, arch: str, dsl: str, impl_info: dict):
        
        # insert into FAISS
        md5_hash = get_md5_hash(impl_code=impl_code)
        file_path = Path(self.island_database_path) / md5_hash
        
        if file_path.exists():
            self.program_list.remove(Program(file_path))

        # import os
        # if os.environ.get('AIKG_DEBUG_MODE', False):
        #     features = json.load(open('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/debug_io/example_output/20c850f9/island_0/metadata.json', 'r'))
        # else:
        features = await self.extract_features(impl_code, framework_code, backend, arch, dsl, '', profile)
            
        file_path.mkdir(parents=True, exist_ok=True)
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)

        impl_file = file_path / "impl_code.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        for vector_store in self.vector_stores:
            vector_store.insert(f"{md5_hash}")
        
        # maintain an online list
        info_data_path = file_path / "impl_info.json"
        with open(info_data_path, 'w', encoding='utf-8') as f:
            json.dump(impl_info, f, ensure_ascii=False, indent=2)
        self.program_list.append(Program(file_path))
        
        # add program to offline evolve database
        src_dir = file_path
        des_dir = Path(self.island_database_path).parent.parent.parent / "evolve_database" / self.evolve_database_suffix
        os.system(f"cp -rf {src_dir} {des_dir}")       
        
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")

    def find_program_by_id(self, id: str) -> Program:
        for p in self.program_list:
            if id == p.get_impl_info()['id']:
                return p
        return None