import json
from typing import Any
from transformers.utils import get_json_schema
from verl.tools.schemas import ToolResponse
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema


class WeatherTool(BaseTool):
    def get_current_temperature(self, city: str):
        """Get current temperature at a location.

        Args:
            city: The location to get the temperature for, in the format "City, State, Country".

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 72,
            "city": city,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}