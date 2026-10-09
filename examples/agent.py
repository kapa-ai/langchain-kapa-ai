"""Run an agent that searches the knowledge base and reads whole documents.

Works with any chat model that supports tool calling. Install that provider's
LangChain package, set its API key, and name the model in LangChain's
provider:model form:

    export KAPA_EXAMPLE_MODEL="<provider>:<model>"
"""

import json
import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_kapa_ai import KapaDocumentsPage, KapaToolkit

MODEL = os.environ.get("KAPA_EXAMPLE_MODEL", "")
if ":" not in MODEL:
    sys.exit(
        "Set KAPA_EXAMPLE_MODEL to a chat model that supports tool calling, in "
        "LangChain's \"<provider>:<model>\" form, and install that provider's "
        "LangChain package."
    )

SYSTEM_PROMPT = (
    "You answer questions from the knowledge base. Search before you answer. "
    "When a chunk is not enough, fetch its whole document with "
    "get_knowledge_documents, passing the chunk's source link unchanged. Cite the "
    "source links of the chunks you used, exactly as the search returned them. "
    "If the knowledge base does not answer the question, say so."
)

search_tool, documents_tool = KapaToolkit().get_tools()

graph = create_agent(
    init_chat_model(MODEL),
    tools=[search_tool, documents_tool],
    system_prompt=SYSTEM_PROMPT,
)


def snippet(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    sentence_end = flat.find(". ")
    if 0 <= sentence_end < limit:
        return flat[: sentence_end + 1]
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def show_search_results(message: ToolMessage) -> None:
    chunks = message.artifact
    if not isinstance(chunks, list):
        print(f"  {snippet(message.text, 300)}")
        return
    if not chunks:
        print("  (no chunks)")
    for number, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, Document):
            source = chunk.metadata["source"] or "(no link)"
            print(f"  [{number}] {source}")
            print(f"      {snippet(chunk.page_content)}")


def show_documents(message: ToolMessage) -> None:
    page = message.artifact
    if not isinstance(page, KapaDocumentsPage):
        print(f"  {snippet(message.text, 300)}")
        return
    if not page.documents:
        print("  (no documents)")
    for document in page.documents:
        length = (
            f"{len(document.content):,} chars"
            if document.content is not None
            else "content unavailable"
        )
        print(f"  {document.source_url or document.document_id}: {length}")
    if page.has_more:
        print(f"  more requested items on page {page.next_page}")


def show(message: BaseMessage) -> None:
    if isinstance(message, AIMessage):
        if message.tool_calls:
            if message.text.strip():
                print(f"\n{message.text.strip()}")
            for call in message.tool_calls:
                print(f"\nTool call: {call['name']} {json.dumps(call['args'])}")
        else:
            print(f"\nAnswer\n------\n{message.text}")
    elif isinstance(message, ToolMessage):
        if message.name == search_tool.name:
            print("Search results:")
            show_search_results(message)
        else:
            print("Documents:")
            show_documents(message)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I get started?"
    print(f"Question: {question}")
    for update in graph.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="updates"
    ):
        for step in update.values():
            for message in step["messages"]:
                show(message)


if __name__ == "__main__":
    main()
