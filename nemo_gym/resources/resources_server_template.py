
from pydantic import BaseModel

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)


class ExampleMultiStepResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ExampleMultiStepResourcesServer(SimpleResourcesServer):
    config: ExampleMultiStepResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    ExampleMultiStepResourcesServer.run_webserver()
