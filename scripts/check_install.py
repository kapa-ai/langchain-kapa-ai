from importlib.metadata import version
from importlib.resources import files

import langchain_kapa_ai
from langchain_kapa_ai import KapaGetDocumentsTool, KapaRetriever, KapaToolkit

assert langchain_kapa_ai.__file__ is not None
assert "site-packages" in langchain_kapa_ai.__file__
assert files("langchain_kapa_ai").joinpath("py.typed").is_file()
assert langchain_kapa_ai.__version__ == version("langchain-kapa-ai")
KapaRetriever(api_key="key", project_id="project")
KapaGetDocumentsTool(api_key="key", project_id="project")
assert len(KapaToolkit(api_key="key", project_id="project").get_tools()) == 2
print(f"langchain-kapa-ai {langchain_kapa_ai.__version__} installs and imports")
