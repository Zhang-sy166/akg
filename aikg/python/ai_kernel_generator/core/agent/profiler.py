# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Tuple
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory, get_md5_hash
logger = logging.getLogger(__name__)


class Profiler(AgentBase):
    """
    使用 NCU 等硬件分析工具测评 impl code 性能
    """

    def __init__(self, model_config: dict, framework: str = "", task_desc: str = "", impl_code: str = "", dsl: str = "", ncu_rep: str = "",
                 optimize_history: str=""):
        self.model_config = model_config
        self.impl_code = impl_code
        self.framework = framework
        self.task_desc = task_desc
        self.dsl = dsl
        self.ncu_rep = ncu_rep
        self.optimize_history = optimize_history

        context = {
            "agent_name": "profiler",
        }
        super().__init__(context=context)

        # 初始化解析器
        self.feature_parser = ParserFactory.get_feature_parser()
        self.format_instructions = self.feature_parser.get_format_instructions()

        # 初始化模板
        self.gen_profile_suggestion_template = self.load_template("profiler/gen_profile_suggestion.j2")

        self.gen_profile_suggestion_input = {
            "op_name": self.impl_code,
            "framework": self.framework,
            "task_desc": self.task_desc,
            "impl_code": self.impl_code,
            "dsl": self.dsl,
            "ncu_profile_res": self.ncu_rep,
        }

    async def run(self) -> Tuple[str, str, str]:
        # 执行LLM生成前更新context，确保正确性
        hash = get_md5_hash(impl_code=self.impl_code)
        to_update_context = {
            "impl_code": self.impl_code,
            "optimize_history": self.optimize_history,
            "hash": hash,
        }
        self.context.update(to_update_context)
        logging.info("NCU Profiler calling LLM ...")
        
        # DEBUG MODE
        import os
        if os.environ.get("AIKG_DEBUG_MODE", False):
            example_res = ''.join(open('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/debug_io/example_output/ncu_profile_res.txt', 'r').readlines())
            return example_res, '', ''
        else:
            return await self.run_llm(self.gen_profile_suggestion_template, self.gen_profile_suggestion_input, self.model_config.get("profiler", "deepseek_r1_default"))
