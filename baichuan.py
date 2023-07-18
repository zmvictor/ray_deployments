from base_model import BaseModel
import logging
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch

logger = logging.getLogger(__name__)

class Baichuan7B(BaseModel):
    def __init__(self):
        self.is_ready = False
        model = "baichuan-inc/Baichuan-7B"
        logger.info(f"Loading {model} model... Start")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True)
        self.llm = self.llm.to("cuda")
        logger.info(f"Loading {model} model... Done")
        self.is_ready = True
        
    @classmethod
    def model_params(self) -> dict:
        return {
            "prompt": "",
            "temperature": 0.5,
            "max_new_tokens": 50,
            "repetition_penalty": 1.1,
        }
    
    def is_healthy(self) -> bool:
        return self.is_ready
    
    def eval(self, *args, **kwargs):
        model_params = self.model_params()
        model_params.update({k:v for k,v in kwargs.items() if k in model_params})
        inputs = self.tokenizer(model_params["prompt"], return_tensors="pt")
        inputs = inputs.to('cuda')
        model_params.pop("prompt")
        pred = self.llm.generate(**inputs, **model_params)
        output = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return output
    

class Baichuan13B(BaseModel):
    def __init__(self):
        self.is_ready = False
        model = "baichuan-inc/Baichuan-13B-Chat"
        logger.info(f"Loading {model} model... Start")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
        self.llm = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.llm = self.llm.to("cuda")
        self.llm.generation_config = GenerationConfig.from_pretrained(model)
        logger.info(f"Loading {model} model... Done")
        self.is_ready = True
        
    @classmethod
    def model_params(self) -> dict:
        return {
            "messages": [{"role": "user", "content": "Hello World!"}],
        }
    
    def is_healthy(self) -> bool:
        return self.is_ready
    
    def eval(self, *args, **kwargs):
        model_params = self.model_params()
        model_params.update({k:v for k,v in kwargs.items() if k in model_params})
        messages = model_params["messages"]
        responses = self.llm.chat(self.tokenizer, messages)
        return responses