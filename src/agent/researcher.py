import math
import numexpr
import re
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from typing import Sequence, Dict

# 定义 calculator 工具
def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.' 
            " Please try again with a valid numerical expression"
        )

calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"

# 定义其他工具
web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# 创建 Researcher 代理
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
researcher_agent = create_react_agent(researcher_llm, tools=tools)

def researcher_node(state):
    result = researcher_agent.invoke(state)
    return {
        "messages": state.messages + [HumanMessage(content=result["messages"][-1].content, name="Researcher")],
        "next": "Supervisor"
    }
