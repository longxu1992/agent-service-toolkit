import json
import os
import httpx
from typing import AsyncGenerator, Dict, Any, Generator
from schema import ChatMessage, UserInput, StreamInput, Feedback


class AgentClient:
    """与代理服务交互的客户端。"""

    def __init__(self, base_url: str = "http://localhost:80", timeout: float | None = None):
        """
        初始化客户端。

        参数：
            base_url (str): 代理服务的基础URL。
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout

    @property
    def _headers(self):
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    async def ainvoke(
        self, message: str, model: str | None = None, thread_id: str | None = None
    ) -> ChatMessage:
        """
        异步调用代理。只返回最终消息。

        参数：
            message (str): 发送给代理的消息
            model (str, 可选): 用于代理的LLM模型
            thread_id (str, 可选): 用于继续对话的线程ID

        返回：
            AnyMessage: 代理的响应
        """
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/invoke",
                json=request.dict(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return ChatMessage.model_validate(response.json())
            else:
                raise Exception(f"Error: {response.status_code} - {response.text}")

    def invoke(
        self, message: str, model: str | None = None, thread_id: str | None = None
    ) -> ChatMessage:
        """
        同步调用代理。只返回最终消息。

        参数：
            message (str): 发送给代理的消息
            model (str, 可选): 用于代理的LLM模型
            thread_id (str, 可选): 用于继续对话的线程ID

        返回：
            ChatMessage: 代理的响应
        """
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        response = httpx.post(
            f"{self.base_url}/invoke",
            json=request.dict(),
            headers=self._headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            return ChatMessage.model_validate(response.json())
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    raise Exception(parsed["content"])

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        同步流式传输代理的响应。

        代理过程的每个中间消息都会作为ChatMessage产生。
        如果stream_tokens为True（默认值），响应还会在生成时产生来自流式模型的内容标记。

        参数：
            message (str): 发送给代理的消息
            model (str, 可选): 用于代理的LLM模型
            thread_id (str, 可选): 用于继续对话的线程ID
            stream_tokens (bool, 可选): 在生成时流式传输标记
                默认值：True

        返回：
            Generator[ChatMessage | str, None, None]: 代理的响应
        """
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        with httpx.stream(
            "POST",
            f"{self.base_url}/stream",
            json=request.dict(),
            headers=self._headers,
            timeout=self.timeout,
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            for line in response.iter_lines():
                if line.strip():
                    parsed = self._parse_stream_line(line)
                    if parsed is None:
                        break
                    yield parsed

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        异步流式传输代理的响应。

        代理过程的每个中间消息都会作为AnyMessage产生。
        如果stream_tokens为True（默认值），响应还会在生成时产生来自流式模型的内容标记。

        参数：
            message (str): 发送给代理的消息
            model (str, 可选): 用于代理的LLM模型
            thread_id (str, 可选): 用于继续对话的线程ID
            stream_tokens (bool, 可选): 在生成时流式传输标记
                默认值：True

        返回：
            AsyncGenerator[ChatMessage | str, None]: 代理的响应
        """
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/stream",
                json=request.dict(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Error: {response.status_code} - {response.text}")
                async for line in response.aiter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: Dict[str, Any] = {}
    ):
        """
        为运行创建反馈记录。

        这是LangSmith create_feedback API的简单封装，因此凭证可以存储和管理在服务中而不是客户端。
        参见：https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/feedback",
                json=request.dict(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            response.json()