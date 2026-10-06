from __future__ import annotations

from typing import Any

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_kapa_ai import KapaRetriever
from tests.integration_tests.live import load_queries, requires_queries

pytestmark = requires_queries

DEEP_COUNT_REASON = (
    "The deep mode returns a variable number of results, capped by top_k, so an "
    "exact count is not part of its contract."
)


class TestKapaRetrieverDefaultMode(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[KapaRetriever]:
        return KapaRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"mode": "default"}

    @property
    def retriever_query_example(self) -> str:
        return load_queries()[0]

    @property
    def num_results_arg_name(self) -> str:
        return "top_k"


class TestKapaRetrieverDeepMode(TestKapaRetrieverDefaultMode):
    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"mode": "deep"}

    @pytest.mark.xfail(reason=DEEP_COUNT_REASON)
    def test_k_constructor_param(self) -> None:
        super().test_k_constructor_param()

    @pytest.mark.xfail(reason=DEEP_COUNT_REASON)
    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        super().test_invoke_with_k_kwarg(retriever)


@pytest.mark.parametrize("mode", ["default", "deep"])
@pytest.mark.parametrize("top_k", [1, 3])
def test_results_never_exceed_top_k(mode: str, top_k: int) -> None:
    documents = KapaRetriever(mode=mode, top_k=top_k).invoke(load_queries()[0])

    assert len(documents) <= top_k
