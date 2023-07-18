import requests
import click
import logging
import json

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

    "Baichuan13B": {
        "model": "Baichuan13B",
        "messages": [{"role": "user", "content": "What's the highest mountain in the world?"}]
    },

    "Longchat13b16k": {
        "model": "Longchat13b16k",
        "prompts": ["Hello, my name is", "The capital of France is"],
        "temperature": 0.5,
        "max_tokens": 50,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_p": 0.9,
        "top_k": -1,
    },
}


@click.command()
@click.option("--list-models", is_flag=True, help="List supported models and corresponding parameters")
@click.option("--endpoint", default="http://localhost:8000", help="service endpoint")
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