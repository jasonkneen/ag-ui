"""Agentic Chat Multimodal — Accepts images, audio, video, and documents."""

from llama_index.llms.openai import OpenAI
from llama_index.protocols.ag_ui.router import get_ag_ui_workflow_router

agentic_chat_multimodal_router = get_ag_ui_workflow_router(
    llm=OpenAI(model="gpt-5.4"),
    system_prompt=(
        "You are a helpful assistant that can analyze images, audio, video, and documents. "
        "Analyze any media the user sends and answer their questions about it. "
        "Be descriptive when analyzing visual content. "
        "If the user sends multiple files, analyze each one."
    ),
)
