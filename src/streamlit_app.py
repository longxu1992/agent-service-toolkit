import asyncio
import os
from typing import AsyncGenerator, List

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatMessage


# 这是一个用于通过简单聊天界面与langgraph代理交互的Streamlit应用。
# 该应用有三个主要的异步运行函数:

# - main() - 设置streamlit应用和高层结构
# - draw_messages() - 绘制一组聊天消息 - 可以是重放现有消息
#   或流式传输新消息。
# - handle_feedback() - 绘制反馈小部件并记录用户反馈。

# 该应用大量使用AgentClient与代理的FastAPI端点进行交互.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"


@st.cache_resource
def get_agent_client():
    agent_url = os.getenv("AGENT_URL", "http://localhost")
    return AgentClient(agent_url)


async def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    models = {
        "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
        "Gemini 1.5 Flash (streaming)": "gemini-1.5-flash",
        "llama-3.1-70b on Groq": "llama-3.1-70b",
    }
    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            use_streaming = st.toggle("Stream results", value=True)

        @st.dialog("Architecture")
        def architecture_dialog():
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # Draw existing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: List[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter():
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if input := st.chat_input():
        messages.append(ChatMessage(type="human", content=input))
        st.chat_message("human").write(input)
        agent_client = get_agent_client()
        if use_streaming:
            stream = agent_client.astream(
                message=input,
                model=model,
                thread_id=get_script_run_ctx().session_id,
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=input,
                model=model,
                thread_id=get_script_run_ctx().session_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new=False,
):
    """
    绘制一组聊天消息 - 可以是重放现有消息或流式传输新消息。

    此函数有额外的逻辑来处理流式令牌和工具调用。
    - 使用占位容器来渲染到达的流式令牌。
    - 使用状态容器来渲染工具调用。跟踪工具输入和输出
      并相应地更新状态容器。

    该函数还需要在会话状态中跟踪最后一条消息容器,
    因为后续消息可以绘制到同一个容器中。这也用于
    在最新的聊天消息中绘制反馈小部件。

    参数:
        messages_aiter: 要绘制的消息的异步迭代器。
        is_new: 消息是否为新消息。
    """

    # 跟踪最后一条消息容器
    last_message_type = None
    st.session_state.last_message = None

    # 用于中间流式令牌的占位符
    streaming_content = ""
    streaming_placeholder = None

    # 遍历消息并绘制它们
    while msg := await anext(messages_agen, None):
        # str消息表示正在流式传输的中间令牌
        if isinstance(msg, str):
            # 如果占位符为空,这是正在流式传输的新消息的第一个令牌。我们需要进行设置。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"意外的消息类型: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # 来自用户的消息,最简单的情况
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # 来自代理的消息是最复杂的情况,因为我们需要
            # 处理流式令牌和工具调用。
            case "ai":
                # 如果我们正在渲染新消息,将消息存储在会话状态中
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果最后一条消息类型不是AI,创建一个新的聊天消息
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 如果消息有内容,将其写出。
                    # 重置流式变量以准备下一条消息。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 为每个工具调用创建一个状态容器,并按ID存储
                        # 状态容器,以确保结果映射到正确的状态容器。
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""工具调用: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("输入:")
                            status.write(tool_call["args"])

                        # 为每个工具调用期望一个ToolMessage。
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if not tool_result.type == "tool":
                                st.error(f"意外的ChatMessage类型: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 如果是新消息,记录消息,并用结果更新正确的
                            # 状态容器
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("输出:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # 如果遇到意外的消息类型,记录错误并停止
            case _:
                st.error(f"意外的ChatMessage类型: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback():
    """绘制反馈小部件并记录用户反馈。"""

    # 检查当前环境是否为生产环境
    env = os.getenv("ENV", "development")
    if env == "production":
        # 在生产环境中禁用反馈功能
        return

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client = get_agent_client()
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs=dict(
                comment="In-line human feedback",
            ),
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())