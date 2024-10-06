import operator

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.researcher import research_node
from agent.supervisor import supervisor_agent, members
from agent.time_agent import time_agent_node
from typing import TypedDict, Annotated, Sequence


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


# 构建 StateGraph，提供 state_schema
agent_graph = StateGraph(state_schema=AgentState)

# 添加节点
agent_graph.add_node("Researcher", research_node)
agent_graph.add_node("TimeAgent", time_agent_node)
agent_graph.add_node("Supervisor", supervisor_agent)

# 设置入口点
agent_graph.add_edge(START, "Supervisor")

for member in members:
    agent_graph.add_edge(member, "Supervisor")
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
agent_graph.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
agent_graph.add_edge(START, "Supervisor")


# 编译代理
research_assistant = agent_graph.compile(
    checkpointer=MemorySaver(),
)

if __name__ == "__main__":
    import asyncio


    async def main():
        # 示例输入
        inputs = {"messages": [HumanMessage(content="请告诉我当前的日期和时间")]}
        async for s in research_assistant.astream(inputs):
            if "__end__" not in s:
                print(s)
                print("----")


    asyncio.run(main())
