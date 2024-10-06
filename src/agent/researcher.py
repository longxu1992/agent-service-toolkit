from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from typing import Sequence, Dict
from agent.tools import calculator

# 定义工具
web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# 创建Researcher代理
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
researcher_agent = create_react_agent(researcher_llm, tools=tools)

def researcher_node(state: Dict[str, Sequence[BaseMessage]]):
    result = researcher_agent.invoke(state)
    return {
        "messages": state.messages + [HumanMessage(content=result["messages"][-1].content, name="Researcher")],
        "next": "Supervisor"
    }
