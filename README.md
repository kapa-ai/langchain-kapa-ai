# langchain-kapa-ai

LangChain integration for [Kapa.ai](https://www.kapa.ai). It gives Python LangChain applications and agents search over a Kapa knowledge base, and whole-document lookup for the sources a search returns.

- `KapaRetriever` returns the most relevant passages for a query as LangChain documents, each with its source link in `metadata["source"]`.
- `KapaGetDocumentsTool` lets an agent fetch whole documents by the source links that search results cite, or by document ID.

## Installation

```bash
pip install langchain-kapa-ai
```

## Usage

Create an API key for your project in the [Kapa platform](https://app.kapa.ai), then set it and the project ID:

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

The [quickstart](https://docs.kapa.ai/frameworks/langchain/quickstart) attaches the retriever to an agent, and the [reference](https://docs.kapa.ai/frameworks/langchain/reference) describes every setting, the document tool, and the errors.

## Examples and evaluations

- [`examples/`](examples/): search without a model, a fixed question-to-answer pipeline, and an agent that searches and follows citations into whole documents.
- [`evals/`](evals/): a fictional product corpus with questions and reference answers, and a runner that measures retrieval and answer quality.

## Development

The repository uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make format-check lint typecheck test
```

`make integration-test` runs the live tests against a project you choose. It reads `KAPA_API_KEY` and `KAPA_PROJECT_ID`, takes its queries from `KAPA_TEST_QUERIES` (one per line) or `tests/integration_tests/queries.local.txt`, and skips without them.

## License

[MIT](LICENSE)
