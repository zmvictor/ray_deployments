# Local Test

## Install Prerequisites 

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create python3.10 env: 
   
   ```
   conda create -n dev_env python=3.10 -y 
   ```
   
   and activate your env:

   ``` 
   conda activate dev_env
   ```
3. Change to current directory and install dependencies:
   
   ```
   pip install -r requirements.txt
   ```

## Start Ray Serve 

```
$  serve run api_ingress:entrypoint
```

This will load and expose models at http://localhost:8000

## Test Models

You can send HTTP request to the local REST API similar as what you did towards OpenAI API.

While there are multiple ways of doing this such as using [curl](https://curl.se/), [postman](https://www.postman.com/), we provide [client.py](./client.py) for convenience. 

### Check Supported models

Use `curl`:
```
$ curl http://localhost:8000/list
```

Or

Use `client.py`: 
```
$ python client.py --list-models

image models:  ['StableDiffusionV2']
large language models:  ['Baichuan7B']
```

### Retrieve from Models

Use `curl` to generete image into `cat.png`:
```
$ curl -X POST -H "Content-Type: application/json" --data '{"model": "StableDiffusionV2", "prompt": "draw a cat on the tree", "height": 512, "width": 512}' http://localhost:8000/imagine --output cat.png
```

Or

Use `client.py`:
```
$ python client.py --model StableDiffusionV2
```

Use `client.py` to access large language models like [baichuan-7b](https://huggingface.co/baichuan-inc/baichuan-7B):
```
$ python clien.py --model Baichuan7B

response:  登鹳雀楼->王之涣
夜雨寄北->李商隐
过零丁洋->文天祥
己亥杂诗(其五)->龚自珍
```
