import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool

from langchain_kapa_ai import KapaGetDocumentsTool, KapaRetriever

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
    init_chat_model(
        os.environ.get("KAPA_EXAMPLE_MODEL", "gpt-5.1"),
        model_provider=os.environ.get("KAPA_EXAMPLE_MODEL_PROVIDER", "openai"),
    ),
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
