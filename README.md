# langchain-kapa-ai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-kapa-ai?label=%20)](https://pypi.org/project/langchain-kapa-ai/#history)
[![PyPI - Downloads](https://static.pepy.tech/badge/langchain-kapa-ai/month)](https://pepy.tech/project/langchain-kapa-ai)

LangChain integration for [Kapa.ai](https://www.kapa.ai). It provides a retriever, a document tool, and a toolkit that returns both as agent tools, so Python LangChain applications and agents can search a Kapa knowledge base and look up whole documents for the sources a search returns.

- `KapaRetriever` returns the most relevant chunks for a query as LangChain documents, each with its source link in `metadata["source"]`.
- `KapaGetDocumentsTool` lets an agent fetch whole documents by the source links that search results cite, or by document ID.
- `KapaToolkit` returns both as agent tools, configured from one set of settings.

## Installation

```bash
pip install langchain-kapa-ai
```

## Usage

The package searches a Kapa project that already has knowledge sources indexed; [Index your first source](https://docs.kapa.ai/getting-started/index-your-first-source) sets one up. Create an API key for that project in the [Kapa platform](https://app.kapa.ai), then set it and the project ID:

```bash
export KAPA_API_KEY="your-api-key"
export KAPA_PROJECT_ID="your-project-id"
```

```python
from langchain_kapa_ai import KapaRetriever

retriever = KapaRetriever()
for document in retriever.invoke("How do I rotate an API key?"):
    print(document.metadata["source"])
    print(document.page_content)
```

`KapaGetDocumentsTool` fetches whole documents by the source links that search results cite:

```python
from langchain_kapa_ai import KapaGetDocumentsTool

tool = KapaGetDocumentsTool()
print(tool.invoke({"urls": ["https://docs.example.com/guide#install"]}))
```

The retriever also works in a chain. This one passes the chunks and their source links to a chat model named in LangChain's `provider:model` form:

```python
import os

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_kapa_ai import KapaRetriever


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {document.metadata['source']}\n{document.page_content}"
        for document in documents
    )


prompt = ChatPromptTemplate.from_template(
    "Answer from the chunks and cite their sources.\n\n"
    "Chunks:\n{context}\n\nQuestion: {question}"
)
chain = (
    {"context": KapaRetriever() | format_documents, "question": RunnablePassthrough()}
    | prompt
    | init_chat_model(os.environ["KAPA_EXAMPLE_MODEL"])
    | StrOutputParser()
)
print(chain.invoke("How do I rotate an API key?"))
```

`KapaToolkit` gives an agent both tools, a search tool that shows each chunk with its source link and the document tool, from one set of settings:

```python
from langchain_kapa_ai import KapaToolkit

tools = KapaToolkit().get_tools()
```

The [agent example](https://docs.kapa.ai/examples/langchain-knowledge-base-search) attaches the toolkit to an agent, and the [reference](https://docs.kapa.ai/retrieval/frameworks/langchain) describes every setting, the document tool, and the errors.

## Examples

- [`examples/`](examples/): search without a model, a fixed question-to-answer pipeline, and an agent that searches and follows citations into whole documents.

The examples work with any chat model that supports tool calling, for example from OpenAI or Anthropic. Install that provider's LangChain package, set its API key, and name the model in LangChain's `provider:model` form:

```bash
export KAPA_EXAMPLE_MODEL="<provider>:<model>"
```

## Development

The repository uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make format-check lint typecheck test
```

`make integration-test` runs the live tests against a project you choose. It reads `KAPA_API_KEY` and `KAPA_PROJECT_ID`, takes its queries from `KAPA_TEST_QUERIES` (one per line) or `tests/integration_tests/queries.local.txt`, and skips without them.

The live example tests also need the LangChain package of the provider named in `KAPA_EXAMPLE_MODEL` available to the project interpreter, so run them with `uv run --with langchain-<provider> pytest tests/integration_tests`.

## Releasing

Releases are published from version tags by the [publish workflow](.github/workflows/publish.yml).

1. Set the version with `uv version <version>` and add a `## <version>` section to [CHANGELOG.md](CHANGELOG.md).
2. Merge the change to `main`.
3. Tag the merged commit with the bare version, for example `git tag 0.1.0 origin/main`, and push the tag.

The workflow checks that the tag matches the package version, the tagged commit is on `main`, and the changelog has the entry. It then runs the quality gates, publishes to TestPyPI and installs the result, and waits for a reviewer to approve the `pypi` environment before publishing to PyPI and installing the published package.

## License

[MIT](LICENSE)
