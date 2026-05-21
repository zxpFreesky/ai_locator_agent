"""
task_planner - 多步任务规划器

将复杂的自然语言指令分解为原子操作序列，逐步执行。

工作流程：
    1. 采集当前页面 DOM 快照
    2. 将指令 + DOM 上下文发送给 LLM，生成结构化的步骤列表
    3. 逐步调用 SmartExecutor 的单步方法执行每个原子操作
    4. 每步执行后等待页面稳定，必要时重新规划后续步骤

步骤格式（LLM 输出）：
    [
        {"action": "fill",  "instruction": "搜索输入框", "value": "关键词"},
        {"action": "click", "instruction": "搜索按钮"},
        {"action": "click", "instruction": "第一个搜索结果"}
    ]
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ai_locator_agent.utils import debug_print, clean_llm_output
from ai_locator_agent.llm_config import get_llm


TASK_PLANNER_SYSTEM_PROMPT = """你是一个 UI 自动化任务规划器。你的任务是将用户的复杂操作指令分解为一系列原子操作步骤。

可用操作类型：
- click: 点击元素（如按钮、链接、标签页）
- fill: 在输入框中填写文本
- select: 在下拉框中选择选项
- hover: 悬停在元素上
- check: 勾选复选框

输出格式（严格 JSON 数组，不要包含其他内容）：
[
    {"action": "fill", "instruction": "搜索输入框", "value": "要填写的文本"},
    {"action": "click", "instruction": "搜索按钮"},
    {"action": "click", "instruction": "第一个搜索结果"}
]

规则：
1. instruction 必须是简洁、明确的元素描述（如"登录按钮"、"用户名输入框"），不要包含操作动词
2. value 仅在 fill 和 select 操作时需要提供
3. 步骤数量不超过 10 步
4. 每个步骤必须是单一、原子操作
5. 如果用户指令本身就是一个单步操作（如"点击登录按钮"），也输出单步数组
6. 输出纯 JSON 数组，不要包含 markdown 代码块标记或其他文字"""

PLANNER_USER_PROMPT_TEMPLATE = """当前页面 DOM 元素摘要（前 40 个可见元素）：
{dom_summary}

用户指令：{instruction}

请将指令分解为原子操作步骤，输出 JSON 数组。"""


class TaskPlanner:
    """
    多步任务规划器

    调用 LLM 将复杂自然语言指令分解为原子操作序列，
    每步由 SmartExecutor 的单步方法执行。
    """

    MAX_STEPS = 10

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    async def plan(
        self, instruction: str, dom_structure: list
    ) -> List[Dict[str, str]]:
        """
        将复杂指令分解为原子操作步骤列表

        Args:
            instruction: 用户自然语言指令
            dom_structure: DOMExtractor 返回的元素列表

        Returns:
            步骤列表，每个步骤是 {"action": str, "instruction": str, "value"?: str}
        """
        truncated = dom_structure[:40]
        dom_summary = json.dumps(truncated, ensure_ascii=False)
        if len(dom_summary) > 6000:
            dom_summary = dom_summary[:6000] + "\n... (截断)"

        user_prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
            dom_summary=dom_summary, instruction=instruction
        )
        messages = [
            SystemMessage(content=TASK_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = clean_llm_output(response.content)
            debug_print(f"[TaskPlanner] LLM 原始输出: {raw[:500]}")
        except Exception as e:
            debug_print(f"[TaskPlanner] LLM 调用失败: {e}")
            return self._fallback_single_step(instruction)

        steps = self._parse_steps(raw)
        if not steps:
            debug_print("[TaskPlanner] 无法解析步骤，降级为单步 click")
            return self._fallback_single_step(instruction)

        steps = steps[: self.MAX_STEPS]
        debug_print(
            f"[TaskPlanner] 分解为 {len(steps)} 个步骤: "
            + " → ".join(s.get("instruction", "?")[:20] for s in steps)
        )
        return steps

    def _parse_steps(self, raw: str) -> Optional[List[Dict[str, str]]]:
        """
        从 LLM 输出中解析 JSON 步骤数组

        容错处理：尝试提取 JSON 数组片段、清理 markdown 标记等。
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        json_start = text.find("[")
        json_end = text.rfind("]")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = text[json_start : json_end + 1]
        try:
            steps = json.loads(json_str)
        except json.JSONDecodeError:
            debug_print(f"[TaskPlanner] JSON 解析失败: {json_str[:200]}")
            return None

        if not isinstance(steps, list) or len(steps) == 0:
            return None

        valid_actions = {"click", "fill", "select", "hover", "check"}
        validated = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action", "")
            instr = step.get("instruction", "")
            if action not in valid_actions or not instr:
                continue
            validated.append(
                {
                    "action": action,
                    "instruction": instr,
                    **({"value": str(step["value"])} if "value" in step else {}),
                }
            )
        return validated if validated else None

    def _fallback_single_step(self, instruction: str) -> List[Dict[str, str]]:
        """
        降级策略：当 LLM 无法分解时，将整个指令作为单个 click 操作
        """
        return [{"action": "click", "instruction": instruction}]

    async def replan_remaining(
        self,
        original_instruction: str,
        completed_steps: List[Dict[str, str]],
        failed_step: Dict[str, str],
        dom_structure: list,
    ) -> Optional[List[Dict[str, str]]]:
        """
        重新规划剩余步骤（当某步失败且页面已变化时调用）

        将已完成步骤和失败信息告知 LLM，让它重新规划后续操作。

        Args:
            original_instruction: 原始用户指令
            completed_steps: 已成功完成的步骤
            failed_step: 失败的步骤
            dom_structure: 当前页面最新的 DOM 快照

        Returns:
            重新规划的步骤列表，或 None 表示放弃
        """
        truncated = dom_structure[:40]
        dom_summary = json.dumps(truncated, ensure_ascii=False)
        if len(dom_summary) > 6000:
            dom_summary = dom_summary[:6000] + "\n... (截断)"

        completed_desc = "\n".join(
            f"  - {s['action']}: {s['instruction']}"
            + (f" (值: {s['value']})" if "value" in s else "")
            for s in completed_steps
        )
        failed_desc = (
            f"{failed_step['action']}: {failed_step['instruction']}"
            + (f" (值: {failed_step['value']})" if "value" in failed_step else "")
        )

        replan_prompt = f"""原始任务：{original_instruction}

已完成的步骤：
{completed_desc if completed_desc else '  (无)'}

失败的步骤：{failed_desc}

当前页面 DOM 元素摘要：
{dom_summary}

请基于当前页面状态，重新规划剩余操作步骤。如果任务已经基本完成，可以输出空数组 []。
输出 JSON 数组。"""

        messages = [
            SystemMessage(content=TASK_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=replan_prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = clean_llm_output(response.content)
            debug_print(f"[TaskPlanner] 重规划输出: {raw[:300]}")
        except Exception as e:
            debug_print(f"[TaskPlanner] 重规划 LLM 调用失败: {e}")
            return None

        steps = self._parse_steps(raw)
        if steps:
            debug_print(
                f"[TaskPlanner] 重规划为 {len(steps)} 个步骤: "
                + " → ".join(s.get("instruction", "?")[:20] for s in steps)
            )
        return steps
