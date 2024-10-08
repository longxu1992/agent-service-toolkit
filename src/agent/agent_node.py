from langchain_core.messages import HumanMessage


def agent_node(state, agent, name):
    agent.invoke(state)
    return {
        "messages": [HumanMessage(content=state["messages"][-1].content, name=name)]
    }
