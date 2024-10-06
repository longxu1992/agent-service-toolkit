from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.supervisor import supervisor_agent
from agent.researcher import researcher_node
from agent.time_agent import time_node
from pydantic import BaseModel
from typing import List

# 定义状态模式
class AgentState(BaseModel):
    messages: List[HumanMessage]
    next: str = "Supervisor"  # 初始化时的默认值

# 构建 StateGraph，提供 state_schema
agent_graph = StateGraph(state_schema=AgentState)

# 添加节点
agent_graph.add_node("Researcher", researcher_node)
agent_graph.add_node("TimeAgent", time_node)
agent_graph.add_node("Supervisor", supervisor_agent)

# 设置入口点
agent_graph.add_edge(START, "Supervisor")

# 从 Supervisor 添加条件边
agent_graph.add_conditional_edges(
    "Supervisor",
    lambda x: x.next,  # 使用属性访问
    {
        "Researcher": "Researcher",
        "TimeAgent": "TimeAgent",
        "FINISH": END
    }
)

# 设置各个 Agent 返回到 Supervisor
agent_graph.add_edge("Researcher", "Supervisor")
agent_graph.add_edge("TimeAgent", "Supervisor")

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
