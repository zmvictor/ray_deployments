from base_model import BaseModel
import logging
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
import torch

logger = logging.getLogger(__name__)

class StableDiffusionV2(BaseModel):
    def __init__(self):
        self.is_ready: bool = False
        
        logger.info("Loading Stable Diffusion V2 model...")
        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        logger.info("Loading Stable Diffusion V2 model... Done")

        logger.info("Moving Stable Diffusion V2 model to GPU...")
        self.pipe = self.pipe.to("cuda")
        logger.info("Moving Stable Diffusion V2 model to GPU... Done")
        
        self.is_ready = True

    @classmethod
    def model_params(self) -> dict:
        return {
            "prompt": "",
            "height": 512,
            "width": 512,
            "num_inference_steps": 50,
            "guidance_scale": 0.75,
        }

    def is_healthy(self) -> bool:
        return self.is_ready
    
    def eval(self, *args, **kwargs):
        model_params = self.model_params()
        model_params.update({k:v for k,v in kwargs.items() if k in model_params})
        image = self.pipe(**model_params).images[0]
        return image
