import os
import json
import logging
from typing import Any, List, Dict
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def _load_embedding_model(embedding_model_name: str = "/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese"):
    """从配置文件加载嵌入模型"""
    print(f"Loading embedding model: {embedding_model_name}")
    
    def load_huggingface_embedding_model(model_name: str):
        """加载HuggingFace嵌入模型"""
        try:
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embedding
        except Exception:
            print(f"Failed to load HuggingFace model: {model_name}")
            return None
    
    embedding = load_huggingface_embedding_model(embedding_model_name)
    if embedding:
        return embedding
    
    print("Automatic download of embedding model failed, using local embedding model")
    value = os.getenv("EMBEDDING_MODEL_PATH")  # 获取环境变量
    if value is None:
        print("EMBEDDING_MODEL_PATH environment variable not set")
    else:
        embedding = load_huggingface_embedding_model(value)
        if embedding:
            return embedding
    return None

embedding_model = _load_embedding_model() 

dummy_doc = Document(page_content="", metadata={})
vector_store = FAISS.from_documents([dummy_doc], embedding_model)
# 删除 dummy 文档
dummy_id = list(vector_store.index_to_docstore_id.values())[0]
vector_store.delete([dummy_id])


def get_page_content(metadata: dict, features: List[str]):
    """
    从算子元数据生成特定特征的页面内容
    page_content为用于向量化存储的文本内容。
    
    Args:
        metadata: 算子元数据字典
        features: 需要提取的特征列表
        
    Returns:
        str: 格式化的特征文本内容
    """
    page_content_parts = []
    for schedule_block in features:
        # 处理schedule块字段，直接展开为键值对
        block = metadata.get('schedule', {}).get(schedule_block, {})
        if isinstance(block, dict):
            for key, value in block.items():
                page_content_parts.append(f"{key}: {value}")
    return ", ".join(page_content_parts)
    
def gen_document(metadata: dict, file_path: str, other_args: Any = None):
    """从算子元数据生成文档，支持schedule块字段"""
    page_content = get_page_content(metadata, ["base", "pass", "text"])
    return Document(
        page_content=page_content,
        metadata={"file_path": file_path}
    )
    

def insert(doc_path: str):
    """向向量存储添加新的文档"""

    metadata_path = Path(doc_path) / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"算子元数据文件 {str(metadata_path)} 不存在")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    op_dir = metadata_path.parent
    doc = gen_document(metadata, str(op_dir))
    # self.delete(doc_path)
    vector_store.add_documents([doc])
    # vector_store.save_local(self.index_path)
    print(f"Successfully added document with path={doc_path} to vector index")
    
print('try to insert ... ')
import time
start = time.time()
for i in range(50):
    insert('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/program_database/Square_matrix_multiplication__1769009520/island_0/42997c4adf374417838724ca508389ba')
end = time.time()
print(f'duration {end-start} s')
import pdb;pdb.set_trace()
