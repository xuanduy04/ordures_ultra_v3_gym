from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import SimpleResourcesServer
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)
from resources_servers.math_with_tool_and_judge.calculator import evaluate


class CalculateRequest(BaseModel):
    expression: str = Field(strict=True, min_length=1, max_length=1000)


class CalculateResponse(BaseModel):
    expression: str
    result: int | float


class MathWithToolAndJudgeConfig(LibraryJudgeMathResourcesServerConfig):
    should_use_judge: bool = False


class MathWithToolAndJudgeServer(LibraryJudgeMathResourcesServer):
    config: MathWithToolAndJudgeConfig

    def setup_webserver(self) -> FastAPI:
        if self.config.should_use_judge:
            app = super().setup_webserver()
        else:
            app = SimpleResourcesServer.setup_webserver(self)

        app.post("/calculate", response_model=CalculateResponse)(self.calculate)
        return app

    async def calculate(self, body: CalculateRequest):
        try:
            result = evaluate(body.expression)
        except (ValueError, ArithmeticError, TypeError, RecursionError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        return CalculateResponse(
            expression=body.expression,
            result=result,
        )


if __name__ == "__main__":
    MathWithToolAndJudgeServer.run_webserver()
