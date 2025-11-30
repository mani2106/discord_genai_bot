from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import TextBlock, ImageBlock
from typing import Dict, List

# Configure vLLM similarly to image_cap_flow/test_img.py
llm = OpenAILike(
    model="/models/qwen3vl_2b",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    request_timeout=360.0,
    context_window=4096,
    is_chat_model=True,
    temperature= 0.35,
    top_p= 0.8,
    top_k= 10,
    repetition_penalty= 1.8,
    presence_penalty= 0.3,
    frequency_penalty= 1.5,
    max_tokens= 150
)

# In-memory conversation state per session (e.g. user id or channel id)
_conversations: Dict[str, List[ChatMessage]] = {}


def _extract_text_from_response(resp) -> str:
    """Extract text safely from different response shapes.

    Handles objects with `.message.content`, dict-like responses, or plain strings.
    """
    try:
        # common wrapper with .message.content
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            content = resp.message.content
            # If content is a TextBlock-like object
            if hasattr(content, "text"):
                return content.text
            # If content is a list of blocks (TextBlock / ImageBlock), extract text blocks
            if isinstance(content, (list, tuple)):
                parts = []
                for item in content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(content)

        # dict-like
        if isinstance(resp, dict):
            # try common keys
            for key in ("message", "content", "text", "response"):
                if key in resp:
                    val = resp[key]
                    if isinstance(val, dict) and "content" in val:
                        return val["content"]
                    return str(val)

        # fallback to stringifying
        return str(resp)
    except Exception:
        return str(resp)


def start_conversation_with_image(session_id: str, image_path: str, prompt: str = "Describe this image.") -> str:
    """
    Read the image, send initial multimodal message, store conversation, and return the assistant reply.
    session_id: string key for user/channel
    """
    with open(image_path, "rb") as f:
        img = f.read()

    final_prompt = f"""
    {prompt}
    Be concise: answer in 1-3 short sentences, do not repeat yourself.
    Also add three or four relevant tags (comma-separated) on a single line.
    """

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(
            role="user",
            content=[
                TextBlock(text=final_prompt),
                ImageBlock(block_type="image", image=img),
            ],
        ),
    ]

    resp = llm.chat(messages)
    reply_text = _extract_text_from_response(resp)

    # persist conversation (system + user + assistant)
    conv = messages.copy()
    # store assistant reply as a TextBlock for consistent conversation shape
    conv.append(ChatMessage(role="assistant", content=TextBlock(text=reply_text)))
    _conversations[session_id] = conv
    return f"""
    Request Prompt: {prompt}\n
    Model answer: {reply_text}
    """


def ask(session_id: str, query: str) -> str:
    """
    Continue conversation for session_id with a text-only user query.
    """
    if session_id not in _conversations:
        return "No image in context. Upload an image first with the /image command."

    prompt = f"""
    Follow-up question about the previous image.
    Be concise: answer in 1-2 short sentences, do not repeat yourself.
    Don't repeat previous answers.
    If the question is not related to the image, respond with 'I can only answer questions related to the image.'
    Question: {query}
    """

    user_msg = ChatMessage(role="user", content=TextBlock(text=prompt))
    # Send the full conversation + new user message
    resp = llm.chat(_conversations[session_id] + [user_msg])
    reply_text = _extract_text_from_response(resp)

    # Persist the new turns
    _conversations[session_id].append(user_msg)
    _conversations[session_id].append(ChatMessage(role="assistant", content=TextBlock(text=reply_text)))
    return query + "\n" + reply_text


def clear(session_id: str) -> None:
    _conversations.pop(session_id, None)
