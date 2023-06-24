from model_handles import image_model_handles, llm_handles

from typing import Dict
from io import BytesIO
from fastapi import FastAPI, status
from fastapi.responses import Response
from starlette.requests import Request

from ray import serve

app = FastAPI()


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, image_handles: Dict[str, serve.Application]) -> None:
        self.image_handles = image_handles


    @app.post(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, request: Request):
        data = await request.json()
        model_name = data.get("model", "")
        if not model_name or model_name not in self.image_handles:
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST, 
                content="Invalid model name"
            )
        self.handle = self.image_handles[model_name]
        image_ref = await self.handle.eval.remote(**data)
        image = await image_ref
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")
    
    @app.get(
        "/",
        status_code=status.HTTP_200_OK,
    )
    async def hc(self, response: Response):
        for name, handle in self.image_handles.items():
            is_ready = await handle.is_healthy.remote()
            if not is_ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    @app.get(
        "/list",
        status_code=status.HTTP_200_OK,
    )
    async def list_models(self):
        return ",".join(list(self.image_handles.keys()))


image_handles = {name: handler.bind() for name, handler in image_model_handles.items()}
entrypoint = APIIngress.bind(image_handles)
