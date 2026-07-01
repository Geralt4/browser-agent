from browser_agent.config import Config
from browser_agent.models.openai_compat import GenericOpenAIAdapter
from browser_agent.perception.vision_router import resolve_use_vision, should_use_vision


class TestShouldUseVision:
    def test_visual_keyword_chart(self):
        assert should_use_vision("describe the chart on this page", True) is True

    def test_visual_keyword_graph(self):
        assert should_use_vision("extract data from the graph", True) is True

    def test_visual_keyword_image(self):
        assert should_use_vision("what image is displayed", True) is True

    def test_visual_keyword_screenshot(self):
        assert should_use_vision("take a screenshot of the dashboard", True) is True

    def test_visual_keyword_layout(self):
        assert should_use_vision("describe the layout of the page", True) is True

    def test_visual_keyword_color(self):
        assert should_use_vision("what color is the button", True) is True

    def test_visual_keyword_look_like(self):
        assert should_use_vision("what does the page look like", True) is True

    def test_visual_keyword_show_me(self):
        assert should_use_vision("show me the navigation menu", True) is True

    def test_visual_keyword_design(self):
        assert should_use_vision("describe the design of the header", True) is True

    def test_visual_keyword_captcha(self):
        assert should_use_vision("solve the captcha on this page", True) is True

    def test_non_visual_navigation(self):
        assert should_use_vision("go to example.com and return the H1", True) is False

    def test_non_visual_form(self):
        assert should_use_vision("fill the form with test data", True) is False

    def test_non_visual_extract(self):
        assert should_use_vision("extract all prices from the table", True) is False

    def test_non_visual_click(self):
        assert should_use_vision("click the submit button", True) is False

    def test_vision_disabled_when_model_doesnt_support(self):
        assert should_use_vision("describe the chart", False) is False

    def test_case_insensitive(self):
        assert should_use_vision("Describe the CHART on this page", True) is True


class TestResolveUseVision:
    def test_auto_mode_visual_task(self):
        cfg = Config(vision_mode="auto")
        adapter = _fake_adapter(supports_vision=True)
        assert resolve_use_vision(cfg, adapter, "describe the chart") is True

    def test_auto_mode_non_visual_task(self):
        cfg = Config(vision_mode="auto")
        adapter = _fake_adapter(supports_vision=True)
        assert resolve_use_vision(cfg, adapter, "go to example.com") is False

    def test_dom_mode_always_off(self):
        cfg = Config(vision_mode="dom")
        adapter = _fake_adapter(supports_vision=True)
        assert resolve_use_vision(cfg, adapter, "describe the chart") is False

    def test_vision_mode_model_supports(self):
        cfg = Config(vision_mode="vision")
        adapter = _fake_adapter(supports_vision=True)
        assert resolve_use_vision(cfg, adapter, "any task") is True

    def test_vision_mode_model_doesnt_support(self):
        cfg = Config(vision_mode="vision")
        adapter = _fake_adapter(supports_vision=False)
        assert resolve_use_vision(cfg, adapter, "any task") is False

    def test_auto_mode_model_doesnt_support(self):
        cfg = Config(vision_mode="auto")
        adapter = _fake_adapter(supports_vision=False)
        assert resolve_use_vision(cfg, adapter, "describe the chart") is False


class TestGenericOpenAIAdapterVision:
    def test_vision_model_in_list(self):
        cfg = Config(vision_models="gpt-4o,llama-3.2-vision", llm_model="gpt-4o", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is True

    def test_vision_model_not_in_list(self):
        cfg = Config(vision_models="gpt-4o,llama-3.2-vision", llm_model="gpt-3.5-turbo", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is False

    def test_vision_models_empty(self):
        cfg = Config(vision_models=None, llm_model="gpt-4o", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is False

    def test_vision_models_blank_string(self):
        cfg = Config(vision_models="", llm_model="gpt-4o", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is False

    def test_vision_models_case_insensitive(self):
        cfg = Config(vision_models="GPT-4o", llm_model="gpt-4o", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is True

    def test_vision_models_with_spaces(self):
        cfg = Config(vision_models=" gpt-4o , llama-3.2-vision ", llm_model="gpt-4o", llm_api_key="sk-test")
        adapter = GenericOpenAIAdapter(cfg)
        assert adapter.supports_vision is True


def _fake_adapter(*, supports_vision: bool):
    class Fake:
        pass
    fake = Fake()
    fake.supports_vision = supports_vision
    return fake


class TestCategoryRouting:
    """Data-driven per-category routing — opt-in via vision_mode="category"."""

    def test_category_mode_routes_dom_categories_to_dom(self):
        cfg = Config(
            vision_mode="category",
            dom_categories="wikipedia,wiki-search",
        )
        adapter = _fake_adapter(supports_vision=True)
        # Listed category → DOM
        assert resolve_use_vision(cfg, adapter, "any task", category="wikipedia") is False
        assert resolve_use_vision(cfg, adapter, "any task", category="wiki-search") is False
        # Unlisted category → vision (because model supports it)
        assert resolve_use_vision(cfg, adapter, "any task", category="navigation") is True

    def test_category_mode_is_case_insensitive(self):
        cfg = Config(
            vision_mode="category",
            dom_categories="Wikipedia,WIKI-SEARCH",
        )
        adapter = _fake_adapter(supports_vision=True)
        assert resolve_use_vision(cfg, adapter, "any task", category="wikipedia") is False
        assert resolve_use_vision(cfg, adapter, "any task", category="Wiki-Search") is False

    def test_category_mode_falls_back_to_auto_when_category_not_supplied(self):
        """A caller that hasn't been refactored to plumb categories through
        should still get sensible behavior. When category is None in
        category mode, fall through to the keyword heuristic."""
        cfg = Config(
            vision_mode="category",
            dom_categories="wikipedia",
        )
        adapter = _fake_adapter(supports_vision=True)
        # Non-visual task with no category: keyword heuristic → False
        assert resolve_use_vision(cfg, adapter, "go to example.com") is False
        # Visual task with no category: keyword heuristic → True
        assert resolve_use_vision(cfg, adapter, "describe the chart") is True

    def test_category_mode_returns_false_when_model_lacks_vision(self):
        cfg = Config(
            vision_mode="category",
            dom_categories="wikipedia",
        )
        adapter = _fake_adapter(supports_vision=False)
        # Unlisted category, but model doesn't support vision → False
        assert resolve_use_vision(cfg, adapter, "any task", category="navigation") is False

    def test_empty_dom_categories_means_all_categories_use_vision(self):
        cfg = Config(vision_mode="category", dom_categories=None)
        adapter = _fake_adapter(supports_vision=True)
        # No categories are listed as DOM, so every category routes to vision.
        assert resolve_use_vision(cfg, adapter, "any task", category="wikipedia") is True
        assert resolve_use_vision(cfg, adapter, "any task", category="multi-step") is True
