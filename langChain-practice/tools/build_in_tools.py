from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("top news in Pakistan today")

print(results)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)


shell_tool = ShellTool()

results = shell_tool.invoke("ls")

print(results)
