from stable_diffusion_v2 import StableDiffusionV2
from ray import serve

image_model_handles = {}

image_model_handles["StableDiffusionV2"] = serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
) (StableDiffusionV2)


# TODO: add text model handles
llm_handles = {}
