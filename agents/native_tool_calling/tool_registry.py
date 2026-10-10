"""
agents/native_tool_calling/tool_registry.py 

Native Tool Registry:
1. Uses Pydantic to automatically generate OpenAI/Groq function calling schemas.
2. Validates incoming arguments form LLMs.
3. Catches runtime exceptions and formats them as observation strings so that LLM can
self-correct.
"""

from matplotlib.style import available
from typing import Callable, Dict, Any, Type, List
import json
from pydantic import BaseModel, ValidationError


class RegisteredTool:
    """
    Encapsulates a Python function, its Pydantic argument schema, and OpenAI tool format.
    """

    def __init__(
            self,
            name: str,
            description: str,
            func: Callable,
            args_schema: Type[BaseModel]
        ):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    
    def to_openai_tool_schema(self) -> Dict[str, Any]:
        """Generates the exact JSON schema dictionary required by Groq and OpenAI."""
        json_schema = self.args_schema.model_json_schema()

        # Clean schema: remove internal Pydantic titles to reduce token overhead
        properties = json_schema.get("properties", {})
        for prop in properties.values():
            prop.pop("title", None)
        
        parameters = {
            "type": "object",
            "properties": properties,
            "required": json_schema.get("required", [])
        }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters
            }
        }
    

    def execute(self, arguments_raw: str) -> str:
        """
        Safely parses arguments, validates then via Pydantic, and calls the function.
        Catches all errors and returns then as strings so the agent does not crash.
        """

        # Step 1: Parse JSON string
        try:
            args_dict = json.loads(arguments_raw) if arguments_raw else {}
        except json.JSONDecodeError as e:
            return f"Tool Execution Error: Failed to parse arguments as JSON: {e}"
        
        # Step 2: Validate against Pydantic schema
        try:
            validated_args = self.args_schema(**args_dict)
        except ValidationError as e:
            return f"Tool Argument Validation Error: {e.errors()}"
        

        # Step 3: Run the underlying Python function
        try:
            result = self.func(**validated_args.model_dump())
            # Convert dict/list results to JSON string; strings remain as is
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)
        except Exception as e:
            return f"Tool Runtime Error in {self.name}: {e}"


class ToolRegistry:
    """Manages a collection of tools and routes tool_calls by name."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}
    
    def register(
            self,
            name: str,
            description: str,
            args_schema: Type[BaseModel]
        ) -> Callable:
        """Decorator to register a tool with its Pydrantic argument model."""
        def decorator(func: Callable) -> Callable:
            tool = RegisteredTool(
                name=name,
                description=description,
                func=func,
                args_schema=args_schema
            )
            self._tools[name] = tool
            return func
        return decorator
    

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns the list of all tool schemas ready to be passed to 
        client.chat.completions.create()
        """
        return [tool.to_openai_tool_schema() for tool in self._tools.values()]
    

    def execute_tool(self, tool_name: str, arguments_raw: str) -> str:
        """Looks up tool by name and executes it safely."""
        if tool_name not in self._tools:
            available = list(self._tools.keys())
            return f"Error: Tool '{tool_name}' does not exist. Available tools: {available}"
        
        return self._tools[tool_name].execute(arguments_raw)

