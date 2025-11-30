from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import TextBlock, ImageBlock

# Configure vLLM with a multimodal model (e.g. LLaVA)
llm = OpenAILike(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    request_timeout=360.0,
    context_window=4096,
    is_chat_model=True,
)

image_path = r"F:\Studies\discord_genai_bot\filestore\images\discord_genai.png"

with open(image_path, "rb") as f:
    img = f.read()

# 3. Multimodal chat (text + image)
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(
        role="user",
        content=[
            TextBlock(text="What's in this picture?"),
            ImageBlock(block_type="image", image=img),
        ])
]

response = llm.chat(messages)
print(response.message.content)