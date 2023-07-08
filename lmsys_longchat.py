from base_model import BaseModel
import logging
from typing import List
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

class Longchat13b16k(BaseModel):
    def __init__(self):
        self.is_ready = False
        model = "lmsys/longchat-13b-16k"
        logger.info(f"Loading {model} model... Start")
        self.llm = LLM(model=model)
        logger.info(f"Loading {model} model... Done")
        self.is_ready = True
        
    @classmethod
    def model_params(self) -> dict:
        return {
            "prompts": [],
            "temperature": 0.5,
            "max_tokens": 50,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 0.9,
            "top_k": -1,
        }
    
    def is_healthy(self) -> bool:
        return self.is_ready
    
    def eval(self, *args, **kwargs):
        model_params = self.model_params()
        model_params.update({k:v for k,v in kwargs.items() if k in model_params})
        prompts = model_params["prompts"]
        model_params.pop("prompts")
        sampling_params = SamplingParams(**model_params)
        outputs = self.llm.generate(prompts, sampling_params)
        output = "\n".join([str(o) for o in outputs])
        return output