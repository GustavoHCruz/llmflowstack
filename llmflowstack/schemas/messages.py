from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


ChatMessages = list[ChatMessage]
