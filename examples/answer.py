import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

from langchain_kapa_ai import KapaRetriever

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
    model = init_chat_model(
        os.environ.get("KAPA_EXAMPLE_MODEL", "gpt-5.1"),
        model_provider=os.environ.get("KAPA_EXAMPLE_MODEL_PROVIDER", "openai"),
    )

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
