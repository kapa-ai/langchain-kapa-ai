"""Run an agent that searches the knowledge base and reads whole documents.

Works with any chat model that supports tool calling. Install that provider's
LangChain package, set its API key, and name the model in LangChain's
provider:model form:

    export KAPA_EXAMPLE_MODEL="<provider>:<model>"
"""

import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool

from langchain_kapa_ai import KapaGetDocumentsTool, KapaRetriever

MODEL = os.environ.get("KAPA_EXAMPLE_MODEL", "")
if ":" not in MODEL:
    sys.exit(
        "Set KAPA_EXAMPLE_MODEL to a chat model that supports tool calling, in "
        "LangChain's \"<provider>:<model>\" form, and install that provider's "
        "LangChain package."
    )

SYSTEM_PROMPT = (
    "You answer questions from the knowledge base. Search before you answer. "
    "When a passage is not enough, fetch its whole document with "
    "kapa_get_documents, passing the passage's source link unchanged. Cite the "
    "source links of the passages you used, exactly as the search returned them. "
    "If the knowledge base does not answer the question, say so."
)

search_tool = create_retriever_tool(
    KapaRetriever(),
    name="search_knowledge_base",
    description=(
        "Search the knowledge base. Returns relevant passages, each with its "
        "source link."
    ),
    document_prompt=PromptTemplate.from_template("Source: {source}\n{page_content}"),
)

graph = create_agent(
    init_chat_model(MODEL),
    tools=[search_tool, KapaGetDocumentsTool()],
    system_prompt=SYSTEM_PROMPT,
)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I get started?"
    for update in graph.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="updates"
    ):
        for step in update.values():
            for message in step["messages"]:
                message.pretty_print()


if __name__ == "__main__":
    main()
