# langchain-kapa-ai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-kapa-ai?label=%20)](https://pypi.org/project/langchain-kapa-ai/#history)
[![PyPI - Downloads](https://static.pepy.tech/badge/langchain-kapa-ai/month)](https://pepy.tech/project/langchain-kapa-ai)

LangChain integration for [Kapa](https://docs.kapa.ai). Kapa is an ingestion and retrieval system: it indexes content such as documentation sites, wikis, PDFs, tickets, Slack threads, and API specifications into a knowledge base, keeps it up to date, and lets your agents get context from it through [agentic retrieval](https://docs.kapa.ai/retrieval/).

This package gives Python LangChain chains and agents access to Kapa's retrieval over the knowledge sources in your Kapa project.

- `KapaRetriever`: a retriever that returns the most relevant chunks for a query as LangChain documents.
- `KapaGetDocumentsTool`: an agent tool that fetches whole documents by source URL or document ID.
- `KapaToolkit`: returns search and document lookup as two agent tools, `search_knowledge_sources` and `get_knowledge_documents`, configured from one set of settings.

## Documentation

- [Reference](https://docs.kapa.ai/retrieval/frameworks/langchain): every setting, the result formats, and the errors.
- [Add knowledge base search to a LangChain agent](https://docs.kapa.ai/examples/langchain-knowledge-base-search): a full walkthrough, from creating an API key to a running agent.

## Requirements

- Python 3.10 or later.
- `langchain-core` 1.4 or later, before 2.0.
- A Kapa project with indexed knowledge sources. To set one up, follow [Set up with a coding agent](https://docs.kapa.ai/getting-started/set-up-with-a-coding-agent).

## Installation

```bash
pip install langchain-kapa-ai
```

## Credentials

In the [Kapa platform](https://app.kapa.ai), create an API key for your project under **Deploy** > **API Keys**, and copy the project ID from **Settings** > **Projects**. All components read both from the environment, so export them:

```bash
export KAPA_API_KEY="your-api-key"
export KAPA_PROJECT_ID="your-project-id"
```

Or put them in a `.env` file. The package does not read the file itself, so load it in your application, for example with [`python-dotenv`](https://pypi.org/project/python-dotenv/):

```bash
KAPA_API_KEY=your-api-key
KAPA_PROJECT_ID=your-project-id
```

## Usage

### Use in an agent

`KapaToolkit` gives an agent both tools. This example also needs `langchain` and your model provider's LangChain package. Replace `<provider>:<model>` with a tool-capable model and configure the provider's credentials.

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_kapa_ai import KapaToolkit

agent = create_agent(
    init_chat_model("<provider>:<model>"),
    tools=KapaToolkit().get_tools(),
    system_prompt="Search the knowledge base before answering and cite your sources.",
)
```

`search_knowledge_sources` returns each chunk with its source link, and `get_knowledge_documents` fetches the whole document behind a link when a chunk is not enough.

### Search without an agent

`KapaRetriever` works anywhere LangChain accepts a retriever, including chains. It supports `invoke`, `ainvoke`, `batch`, and `abatch`:

```python
from langchain_kapa_ai import KapaRetriever

retriever = KapaRetriever()
for document in retriever.invoke("How do I rotate an API key?"):
    print(document.metadata["source"])
    print(document.page_content)
```

#### Retrieval mode

The retriever has two modes. `default` is faster. `deep` searches further and returns only relevant chunks, at the cost of latency. Set the mode on the retriever, or for a single call:

```python
retriever = KapaRetriever(mode="deep")
documents = retriever.invoke("How do I rotate an API key?", mode="default")
```

[Choose a retrieval mode](https://docs.kapa.ai/retrieval/guides/tuning-knowledge-base-search) helps you pick one. `KapaToolkit` takes the same setting, for example `KapaToolkit(mode="deep")`. The [reference](https://docs.kapa.ai/retrieval/frameworks/langchain) lists every other setting, such as `top_k`, `source_group_ids`, and analytics attribution.

### Fetch whole documents

`KapaGetDocumentsTool` fetches whole documents by the source links that search results cite:

```python
from langchain_kapa_ai import KapaGetDocumentsTool

tool = KapaGetDocumentsTool()
print(tool.invoke({"urls": ["https://docs.example.com/guide"]}))
```

### Errors and retries

A failed request raises an exception that derives from `KapaError`; it never becomes an empty result. No component retries a failed request: in a chain, wrap a component with LangChain's `with_retry()`, and in an agent, use LangChain's `ToolRetryMiddleware`. The [reference](https://docs.kapa.ai/retrieval/frameworks/langchain#errors) lists every exception.

## Examples

See [`examples/`](examples/) for full code examples.

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
