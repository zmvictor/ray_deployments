import requests

prompt = "a big tiger is dancing on the tree with a lot of bats"
r = f"http://127.0.0.1:8000/imagine"
# r = f"http://demo.inferinite.ai/imagine"
resp = requests.post(r, json={'model': 'StableDiffusionV2', 'prompt': prompt, 'height': 512, 'width': 512, 'num_inference_steps': 100})
# print(resp.status_code, resp.content)
with open("output.png", 'wb') as f:
    f.write(resp.content)
