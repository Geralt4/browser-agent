import pytest

from browser_agent.config import Config
from browser_agent.models.kimi import KIMI_MODEL, MOONSHOT_BASE_URL, KimiAdapter
from browser_agent.models.openai_compat import GenericOpenAIAdapter
from browser_agent.models.registry import get_adapter


def test_kimi_adapter_builds_chat_model():
    adapter = KimiAdapter(Config(provider="kimi", moonshot_api_key="test-key"))
    assert adapter.supports_vision is True
    assert adapter.name == KIMI_MODEL
    model = adapter.chat_model()
    assert model.model == KIMI_MODEL
    assert model.base_url == MOONSHOT_BASE_URL


def test_kimi_adapter_requires_key():
    with pytest.raises(ValueError):
        KimiAdapter(Config(provider="kimi", moonshot_api_key=None))


def test_generic_adapter_builds_chat_model():
    adapter = GenericOpenAIAdapter(
        Config(
            provider="openai",
            llm_model="gpt-4o-mini",
            llm_api_key="sk-test",
            llm_base_url="https://api.example.com/v1",
        )
    )
    assert adapter.supports_vision is False
    model = adapter.chat_model()
    assert model.model == "gpt-4o-mini"
    assert model.base_url == "https://api.example.com/v1"


def test_registry_selects_provider():
    kimi = get_adapter(Config(provider="kimi", moonshot_api_key="k"))
    generic = get_adapter(Config(provider="openai", llm_model="m", llm_api_key="k"))
    assert isinstance(kimi, KimiAdapter)
    assert isinstance(generic, GenericOpenAIAdapter)


def test_registry_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_adapter(Config(provider="nope"))
