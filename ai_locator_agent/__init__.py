#!/usr/bin/env python3
"""
AI Locator Agent - AI驱动的智能元素定位与多步任务规划
四级降级策略：Level 0 (OCR) → Level 1 (文本) → Level 2 (截图LLM) → Level 3 (坐标)
"""

from ai_locator_agent.utils import debug_print, sanitize_filename
from ai_locator_agent.dom_extractor import DOMExtractor
from ai_locator_agent.llm_config import get_llm, MODEL_CONFIGS
from ai_locator_agent.screenshot_locator import ScreenshotLocator
from ai_locator_agent.locator_agent import LocatorAgent
from ai_locator_agent.visual_locator import VisualLocator
from ai_locator_agent.safe_locator import safe_build_locator
from ai_locator_agent.local_vision import LocalVisionLocator
from ai_locator_agent.executor import SmartExecutor
from ai_locator_agent.task_planner import TaskPlanner
from ai_locator_agent import server
from ai_locator_agent.server import ensure_browser, main

__version__ = "6.0.0"
__all__ = [
    "debug_print", "sanitize_filename",
    "DOMExtractor", "get_llm", "MODEL_CONFIGS",
    "ScreenshotLocator", "LocatorAgent", "VisualLocator",
    "safe_build_locator", "LocalVisionLocator", "SmartExecutor",
    "TaskPlanner", "ensure_browser", "main",
]
