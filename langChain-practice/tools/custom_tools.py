from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type


# method 1: Using the @tool decorator to create a custom tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


result = multiply.invoke({"a": 3, "b": 5})

print(f"@tool Decorator Result: {result}")

print(f"Name: {multiply.name}")
print(f"Description: {multiply.description}")
print(f"Arguments: {multiply.args}")

print(f"Argument Schema: {multiply.args_schema.model_json_schema()}")

print("**" * 20)


# Method 2 - Using StructuredTool
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


multiply_tool = StructuredTool.from_function(
    multiply_func,
    name="multiply_tool",
    description="Multiply two numbers",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 3, "b": 3})

print(f"StructuredTool Result: {result}")
print(f"Name: {multiply_tool.name}")
print(f"Description: {multiply_tool.description}")
print(f"Arguments: {multiply_tool.args}")

print("**" * 20)

# Method 3 - Using BaseTool Class
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool_instance = MultiplyTool()

result = multiply_tool_instance.invoke({"a": 4, "b": 5})

print(f"BaseTool Result: {result}")
print(f"Name: {multiply_tool_instance.name}")
print(f"Description: {multiply_tool_instance.description}")

print("**" * 20)
