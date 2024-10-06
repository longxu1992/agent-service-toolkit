import functools
from datetime import datetime

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.agent_node import agent_node


def get_current_date(input: str) -> str:
    """返回当前日期，格式为YYYY-MM-DD"""
    return datetime.now().strftime('%Y-%m-%d')


def get_current_time(input: str) -> str:
    """返回当前时间，格式为HH:MM:SS"""
    return datetime.now().strftime('%H:%M:%S')


date_tool: BaseTool = tool(get_current_date)
date_tool.name = "get_current_date"
time_tool: BaseTool = tool(get_current_time)
time_tool.name = "get_current_time"

tools = [date_tool, time_tool]

time_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
time_agent = create_react_agent(time_llm, tools=tools)
time_agent_node = functools.partial(agent_node, agent=time_agent, name="TimeAgent")
