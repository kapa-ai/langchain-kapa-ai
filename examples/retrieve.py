import sys

from langchain_kapa_ai import KapaRetriever


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I get started?"
    retriever = KapaRetriever()
    for number, document in enumerate(retriever.invoke(query), start=1):
        print(f"[{number}] {document.metadata['source']}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()
