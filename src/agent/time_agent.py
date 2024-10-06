def time_node(state):
    # TimeAgent节点的实现
    from datetime import datetime
    from langchain_core.messages import HumanMessage, BaseMessage
    from langchain_core.tools import tool, BaseTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from typing import Sequence, Dict

    def get_current_date(input: str) -> str:
        """返回当前日期，格式为YYYY-MM-DD"""
        return datetime.now().strftime('%Y-%m-%d')

    def get_current_time(input: str) -> str:
        """返回当前时间，格式为HH:MM:SS"""
        return datetime.now().strftime('%H:%M:%S')

    current_date_tool: BaseTool = tool(get_current_date)
    current_date_tool.name = "GetCurrentDate"

    current_time_tool: BaseTool = tool(get_current_time)
    current_time_tool.name = "GetCurrentTime"

    # 创建 Time Agent
    time_tools = [current_date_tool, current_time_tool]

    time_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    time_agent = create_react_agent(time_llm, tools=time_tools)

    result = time_agent.invoke(state)
    return {
        "messages": state.messages + [HumanMessage(content=result["messages"][-1].content, name="TimeAgent")],
        "next": "Supervisor"
    }
