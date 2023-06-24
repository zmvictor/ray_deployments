import requests

prompt = "a big tiger is dancing on the tree with a lot of bats"
input = "%20".join(prompt.split(" "))
r = f"http://127.0.0.1:8000/imagine"
# r = f"http://demo.inferinite.ai/sd/imagine"
resp = requests.post(r, json={'model': 'StableDiffusionV2', 'prompt': prompt, 'img_size': 512})
# print(resp.status_code, resp.content)
with open("output.png", 'wb') as f:
    f.write(resp.content)
