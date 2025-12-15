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
    def __init__(self, island_id: int, pd_path: str, database_config: dict):
        self.island_database_path = pd_path + '/island_' + str(island_id)
        import os
        os.makedirs(self.island_database_path, exist_ok=True)
        
        self.base_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='base_vector_store',
            features=['base'],
            config=database_config
        )
        self.pass_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='pass_vector_store',
            features=['pass'],
            config=database_config
        )
        self.text_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='text_vector_store',
            features=['text'],
            config=database_config
        )
        self.vector_stores = [self.base_vector_store, self.pass_vector_store, self.text_vector_store]
        self.vector_store_map = {
            self.base_vector_store: 'base',
            self.pass_vector_store: 'pass',
            self.text_vector_store: 'text',
        }
        
        # maintain an online list
        self.program_list: list[Program] = []
        
        super().__init__(self.island_database_path, self.vector_stores, database_config)
        
        logger.info(f'island {island_id} was created\n')
    
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

        import os
        if os.environ.get('AIKG_DEBUG_MODE', False):
            features = json.load(open('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/debug_io/example_output/20c850f9/island_0/metadata.json', 'r'))
        else:
            features = await self.extract_features(impl_code, framework_code, profile, backend, arch, dsl)
            
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
        
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")

    def find_program_by_id(self, id: str) -> Program:
        for p in self.program_list:
            if id == p.get_impl_info()['id']:
                return p
        return None