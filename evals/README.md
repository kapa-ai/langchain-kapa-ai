# Evaluations

These evaluations measure retrieval and answer quality separately from connector correctness, which the tests cover. Every run is credentialed and runs on your machine; nothing here runs in CI.

## Corpus

`corpus/` holds documents about Fernwick Relay, a fictional webhook relay, so every fact has a known source. Index them into a Kapa project of your choice. The datasets expect the links below, which are fictional and use a reserved example domain.

| File | Link | How to index it | Citation it produces |
| --- | --- | --- | --- |
| `fernwick_client.py` | `https://code.example.com/fernwick/fernwick_client.py` | File Upload, with this **Linked URL** | line anchors, such as `#L12-L16` |
| `fernwick-operations-manual.pdf` | `https://docs.example.com/fernwick/operations-manual.pdf` | File Upload, with this **Linked URL** | page anchors, such as `#page=2` |
| `fernwick-release-handbook.md` | `https://docs.example.com/fernwick/handbook#v2` | File Upload, with this **Linked URL** | the stored link itself, fragment included |
| `fernwick-guide.html` | `https://docs.example.com/fernwick/guide.html` | A website source that serves the page | heading anchors, such as `#retry-policy` |

File Upload does not accept HTML files, so the guide needs a website source. Its citations then use the address the page is served from rather than the link above, so cases that expect the guide count as missed evidence until their `expected_sources` name that address.

`build_corpus_pdf.py` regenerates the PDF from its text.

The `kapa-docs` datasets ask about Kapa's own documentation, so they run against a project that indexes docs.kapa.ai.

## Datasets

Each line of a dataset is one case: the question, its category (`direct`, `paraphrase`, `multi_passage`, `ambiguous`, or `unanswerable`), the expected source links, text the retrieved passages should contain, and a reference answer. Only the grader sees the reference answer.

Develop against `dev.jsonl`. The `heldout.jsonl` sets stay untouched until you report a result, and the runner refuses them without `--allow-heldout`.

## Running

```bash
export KAPA_API_KEY="..." KAPA_PROJECT_ID="..." OPENAI_API_KEY="..."
make eval ARGS="run --dataset evals/datasets/fictional/dev.jsonl --repeats 3"
```

The runner compares the `default` and `deep` modes with every other setting held fixed. Use `--system agent` to evaluate the agent, which chooses its own searches and document lookups, instead of the fixed pipeline. Use `--project-id` when the datasets of one run live in different projects. The answer model and grader default to OpenAI `gpt-5.1`; change them with `--model`, `--model-provider`, `--grader-model`, and `--grader-model-provider`.

Each run writes a folder under `results/` (not committed) with `run.json` (package versions, corpus revision, dataset hashes, and model settings), `cases.jsonl` (retrieved passages, answer, tool calls, latency, call counts, and grades per case), and `summary.json` (results per mode).

## Measures

- Expected evidence retrieval: the share of expected sources and expected text found in the retrieved passages.
- Passage relevance: the share of retrieved passages the grader marks relevant.
- Answer correctness and claim support: graded against the reference answer and the retrieved passages respectively, so an answer that matches the reference with facts absent from the passages is caught.
- Citations: whether every cited link came from the retrieved passages.
- Honest uncertainty: whether the answer says when the sources do not cover the question.
- Tool choices, model calls, and latency.

## Checking the grader

The grader is a model, so check it against your own judgments before relying on it. Write one JSON line per attempt you review into `judgments/`, with `case_id`, `mode`, `repeat`, and any of `correctness` (`correct`, `partial`, or `incorrect`), `claim_support` (`supported`, `partial`, or `unsupported`), and `honest_uncertainty` (`true` or `false`), then compare:

```bash
make eval ARGS="calibrate --run evals/results/<run> --judgments evals/judgments/<file>.jsonl"
```

Review failures by hand, and repeat model-based runs to see how much they vary. A small fictional dataset shows how the connector and an agent behave; it does not establish retrieval quality in general.
