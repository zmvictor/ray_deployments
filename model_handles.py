from stable_diffusion_v2 import StableDiffusionV2
from baichuan import Baichuan7B, Baichuan13B
from lmsys_longchat import Longchat13b16k
from ray import serve

"""
Text to Image models
"""
image_model_handles = {}

image_model_handles["StableDiffusionV2"] = serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
) (StableDiffusionV2)


"""
Large Language Models
"""
large_language_model_handles = {}

# large_language_model_handles["Baichuan7B"] = serve.deployment(
#     ray_actor_options={"num_gpus": 1},
#     autoscaling_config={"min_replicas": 1, "max_replicas": 2},
# ) (Baichuan7B)

large_language_model_handles["Longchat13b16k"] = serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
) (Longchat13b16k)

large_language_model_handles["Baichuan13B"] = serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
) (Baichuan13B)

