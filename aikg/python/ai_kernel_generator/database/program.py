import json
from pathlib import Path


class Program:
    def __init__(self, file_dir: str):
        self.file_dir = file_dir
    
    def get_impl_code(self) -> str:
        return ''.join(open(str(Path(self.file_dir) / 'impl_code.py'), 'r').readlines())
    
    def get_impl_info(self) -> dict:
        return json.load(open(str(Path(self.file_dir) / 'impl_info.json'), 'r'))
    
    def get_impl_feat(self) -> dict:
        return json.load(open(str(Path(self.file_dir) / 'metadata.json'), 'r'))

    def __eq__(self, other):
        return other.file_dir == self.file_dir
        