from __future__ import annotations

from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter

VISUAL_KEYWORDS = [
    "chart",
    "graph",
    "plot",
    "diagram",
    "image",
    "picture",
    "screenshot",
    "layout",
    "color",
    "colour",
    "visual",
    "appearance",
    "look like",
    "show me",
    "what does",
    "compare",
    "difference",
    "render",
    "canvas",
    "drawing",
    "icon",
    "button style",
    "heatmap",
    "design",
    "theme",
    "font",
    "logo",
    "banner",
    "background",
    "animation",
    "video",
    "photo",
    "thumbnail",
    "gallery",
    "slideshow",
    "carousel",
    "map",
    "dashboard",
    "widget",
    "panel",
    "popup",
    "modal",
    "overlay",
    "tooltip",
    "dropdown",
    "menu",
    "navbar",
    "sidebar",
    "footer",
    "header",
    "captcha",
    "qr code",
    "barcode",
]


def should_use_vision(task: str, model_supports_vision: bool) -> bool:
    """Decide whether to enable vision for this task.

    Returns True only if the model supports vision AND the task
    contains keywords suggesting visual understanding is needed.
    """
    if not model_supports_vision:
        return False
    task_lower = task.lower()
    return any(kw in task_lower for kw in VISUAL_KEYWORDS)


def resolve_use_vision(
    cfg: Config,
    adapter: ModelAdapter,
    task: str,
    *,
    category: str | None = None,
) -> bool:
    """Resolve the effective use_vision flag for a task.

    - "dom"      → always False (DOM-first)
    - "vision"   → adapter.supports_vision (always use vision if model supports it)
    - "category" → vision, except when `category` is in cfg.dom_category_list
                   (data-driven per-category routing)
    - "auto"     → per-task heuristic (keyword-based)

    `category` is required for the "category" mode; if it's not supplied
    the mode falls back to "auto" behavior so a caller that hasn't been
    refactored to plumb categories through still works.
    """
    if cfg.vision_mode == "vision":
        return adapter.supports_vision
    if cfg.vision_mode == "dom":
        return False
    if cfg.vision_mode == "category":
        if category is None:
            return should_use_vision(task, adapter.supports_vision)
        if category.lower() in [c.lower() for c in cfg.dom_category_list]:
            return False
        return adapter.supports_vision
    return should_use_vision(task, adapter.supports_vision)
