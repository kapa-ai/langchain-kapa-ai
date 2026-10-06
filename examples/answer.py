"""Answer a question from retrieved passages with a fixed pipeline.

Works with any chat model that supports tool calling. Install that provider's
LangChain package, set its API key, and name the model in LangChain's
provider:model form:

    export KAPA_EXAMPLE_MODEL="<provider>:<model>"
"""

import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

from langchain_kapa_ai import KapaRetriever

MODEL = os.environ.get("KAPA_EXAMPLE_MODEL", "")
if ":" not in MODEL:
    sys.exit(
        "Set KAPA_EXAMPLE_MODEL to a chat model that supports tool calling, in "
        "LangChain's \"<provider>:<model>\" form, and install that provider's "
        "LangChain package."
    )

INSTRUCTIONS = (
    "Answer the question using only the numbered passages. Cite the passages you "
    "used by their source links. If the passages do not answer the question, say "
    "that the knowledge base does not cover it."
)


def format_evidence(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{number}] Source: {document.metadata['source']}\n{document.page_content}"
        for number, document in enumerate(documents, start=1)
    )


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I get started?"
    model = init_chat_model(MODEL)

    documents = KapaRetriever().invoke(question)
    evidence = format_evidence(documents)
    print("Evidence\n--------")
    print(evidence or "(no passages returned)")

    answer = model.invoke(
        [
            ("system", INSTRUCTIONS),
            ("human", f"Passages:\n\n{evidence}\n\nQuestion: {question}"),
        ]
    )
    print("\nAnswer\n------")
    print(answer.text)


if __name__ == "__main__":
    main()
