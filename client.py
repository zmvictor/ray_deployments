import requests
import click
import logging
import json

# prompt = "a big tiger is dancing on the tree with a lot of bats"
# r = f"http://127.0.0.1:8000/llm"
# r = f"http://127.0.0.1:8000/imagine"
# r = f"http://demo.inferinite.ai/imagine"
# resp = requests.post(r, json={'model': 'StableDiffusionV2', 'prompt': prompt, 'height': 512, 'width': 512, 'num_inference_steps': 100})
# resp = requests.post(r, json={'model': 'Baichuan7B', 'prompt': "登鹳雀楼->王之涣"})
# print(resp.status_code, resp.content.decode('utf-8'))
# with open("output.png", 'wb') as f:
#     f.write(resp.content)

logger = logging.getLogger(__name__)

image_models = {
    "StableDiffusionV2" : {
        "model": "StableDiffusionV2",
        "prompt" : "a big tiger is dancing on the tree with a lot of bats",
        "height": 512,
        "width": 512,
        "num_inference_steps": 100,
    },
}

llms = {
    "Baichuan7B": {
        "model": "Baichuan7B",
        "prompt": "登鹳雀楼->王之涣\n夜雨寄北->",
        "temperature": 0.5,
        "max_new_tokens": 100,
    },
}

def print_pretty_content(content: bytes):
    json_object = json.loads(content.decode('utf-8'))
    print(json.dumps(json_object, indent=4))


@click.command()
@click.option("--list-models", is_flag=True, help="List supported models and corresponding parameters")
@click.option("--endpoint", default="http://localhost:8000", help="ray endpoint")
@click.option("--model", help="Model name: must be contained in --list-models")
def client(list_models, endpoint, model):
    if list_models:
        print("image models: ", list(image_models.keys()))
        print("large language models: ", list(llms.keys()))
    elif model in image_models:
        resp = requests.post(f"{endpoint}/imagine", json=image_models[model])
        with open("output.png", 'wb') as f:
            f.write(resp.content)
        print("Generate image at output.png")
    elif model in llms:
        resp = requests.post(f"{endpoint}/llm", json=llms[model])
        print("response: ", resp.content.decode('utf-8'))
    else:
        raise ValueError(f"{model} is not supported. Check supported models via --list-models ")

if __name__ == "__main__":
    client()
        
        