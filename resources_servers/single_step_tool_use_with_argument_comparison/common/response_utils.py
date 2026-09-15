
from typing import Optional, Union

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputText


def extract_tool_call_or_text(
    response: NeMoGymResponse,
) -> Optional[Union[NeMoGymResponseFunctionToolCall, NeMoGymResponseOutputText]]:
    result = None
    for output_item in response.output:
        if output_item.type == "function_call":
            return output_item

        elif output_item.type == "message" and output_item.role == "assistant" and result is None:
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    result = content_item
                    break

    return result
