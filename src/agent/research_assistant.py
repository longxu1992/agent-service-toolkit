from datetime import datetime
import os
import operator
from typing import Annotated, Sequence, Literal
from functools import partial
from typing_extensions import TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from agent.tools import calculator

# 定义AgentState
class AgentState(TypedDict):
    # 添加注解，表示消息列表会被追加
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# 定义工具
web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# 创建Researcher代理
researcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
researcher_agent = create_react_agent(researcher_llm, tools=tools)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

researcher_node = partial(agent_node, agent=researcher_agent, name="Researcher")

# 定义Supervisor代理
members = ["Researcher"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal[tuple(options)]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor_agent(state):
    supervisor_chain = prompt | supervisor_llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)

# 构建StateGraph
agent_graph = StateGraph(AgentState)

# 添加节点
agent_graph.add_node("Researcher", researcher_node)
agent_graph.add_node("supervisor", supervisor_agent)

# 设置入口点
agent_graph.add_edge(START, "supervisor")

# 从supervisor添加条件边
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
agent_graph.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# 设置Researcher返回到supervisor
agent_graph.add_edge("Researcher", "supervisor")

# 编译代理
research_assistant = agent_graph.compile(
    checkpointer=MemorySaver(),
)

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage

    async def main():
        # 示例输入
        inputs = {"messages": [HumanMessage(content="帮我查找巧克力曲奇的食谱")]}
        async for s in research_assistant.astream(
            inputs,
            config=RunnableConfig(),
        ):
            if "__end__" not in s:
                print(s)
                print("----")

    asyncio.run(main())