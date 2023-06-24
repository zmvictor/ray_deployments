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

    def is_healthy(self) -> bool:
        return self.is_ready
    
    def eval(self, *args, **kwargs):
        prompt = kwargs.get("prompt", "")
        img_size = int(kwargs.get("img_size", 512))
        return self._eval(prompt=prompt, img_size=img_size)

    def _eval(self, prompt: str, img_size: int):
        assert len(prompt), "prompt parameter cannot be empty"

        image = self.pipe(prompt, height=img_size, width=img_size).images[0]
        return image