# Changelog

All notable changes to this package are documented here. Versions follow [semantic versioning](https://semver.org).

## 0.1.0

- `KapaRetriever`: synchronous and asynchronous retrieval from a Kapa project in the `default` or `deep` mode, with result limits, source group filters, and analytics options. Each document keeps its citation link in `metadata["source"]`.
- `KapaGetDocumentsTool`: whole-document lookup by source link or document ID, with exact-first link matching, a fallback without the fragment for unmatched links, paging over the requested items, and bounded document content.
- `KapaToolkit`: the search tool and the document tool for an agent, configured from one set of settings.
- Distinct exceptions for authentication, validation, rate limit, service, connection, and response failures.
