
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.response_utils import (
    extract_tool_call_or_text,
)
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedAction,
    StepRewardCategory,
    ToolCallComparator,
    ToolCallComparatorConfig,
)


class SingleStepToolUseArgumentComparisonResourcesServerConfig(BaseResourcesServerConfig):
    tool_call_comparator_config: ToolCallComparatorConfig


class SingleStepToolUseArgumentComparisonRunRequest(BaseRunRequest):
    expected_action: ExpectedAction


class SingleStepToolUseArgumentComparisonVerifyRequest(
    SingleStepToolUseArgumentComparisonRunRequest, BaseVerifyRequest
):
    pass


class SingleStepToolUseArgumentComparisonVerifyResponse(BaseVerifyResponse):
    expected_action: ExpectedAction
    category: StepRewardCategory


class SingleStepToolUseArgumentComparisonResourcesServer(SimpleResourcesServer):
    config: SingleStepToolUseArgumentComparisonResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(
        self, body: SingleStepToolUseArgumentComparisonVerifyRequest
    ) -> SingleStepToolUseArgumentComparisonVerifyResponse:
        extracted_content = extract_tool_call_or_text(body.response)
        if extracted_content is None:
            return SingleStepToolUseArgumentComparisonVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                category=StepRewardCategory.NO_ACTION_FOUND,
            )

        expected_action = body.expected_action
        match expected_action.type:
            case "function_call":
                if extracted_content.type == "function_call":
                    tool_call_comparator = ToolCallComparator(config=self.config.tool_call_comparator_config)
                    reward, category = tool_call_comparator.compare_tool_call(expected_action, extracted_content)

                else:
                    reward = 0.0
                    category = StepRewardCategory.NO_EXPECTED_TOOL_CALL

            case "message":
                if extracted_content.type == "output_text":
                    # Currently, any chat message is assigned a reward of one.
                    reward = 1.0
                    category = StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND

                else:
                    reward = 0.0
                    category = StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE

            case _:
                raise NotImplementedError

        return SingleStepToolUseArgumentComparisonVerifyResponse(
            **body.model_dump(),
            reward=reward,
            category=category,
        )


if __name__ == "__main__":
    SingleStepToolUseArgumentComparisonResourcesServer.run_webserver()
