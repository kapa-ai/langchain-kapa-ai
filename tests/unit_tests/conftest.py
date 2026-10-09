import pytest

from tests.unit_tests.fake_kapa import FakeKapa


@pytest.fixture
def kapa() -> FakeKapa:
    return FakeKapa()
