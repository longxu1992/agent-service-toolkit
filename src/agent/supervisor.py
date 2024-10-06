from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal, Sequence

class RouteResponse(BaseModel):
    next: Literal["Researcher", "TimeAgent", "FINISH"]

members = ["Researcher", "TimeAgent"]
options = ["FINISH"] + members

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    "following workers: {members}. Given the following user request, "
    "respond with the worker to act next. Each worker will perform a "
    "task and respond with their results and status. When finished, "
    "respond with FINISH."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next? "
            "Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_llm = ChatOpenAI(model="gpt-4o-mini")

def supervisor_agent(state):
    # Supervisor节点的实现
    supervisor_chain = prompt | supervisor_llm.with_structured_output(RouteResponse)
    # 只传入 messages
    result = supervisor_chain.invoke(state.messages)
    # 更新状态中的 next
    state.next = result.next
    return state
