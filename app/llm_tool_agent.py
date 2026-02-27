"""LLM-driven orchestration layer: model chooses when to call tools."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from app.tool_system import TelecomToolSystem

ChatFn = Callable[[list[dict[str, str]]], str]


@dataclass
class LLMToolAgentConfig:
    max_tool_rounds: int = 1


class LLMToolAgent:
    """Runs one chat turn with optional tool calls selected by the LLM."""

    def __init__(
        self,
        tool_system: TelecomToolSystem,
        config: LLMToolAgentConfig | None = None,
    ) -> None:
        self.tool_system = tool_system
        self.config = config or LLMToolAgentConfig()

    def run_turn(
        self,
        messages: list[dict[str, str]],
        phone: str,
        state: dict[str, Any] | None,
        llm_chat: ChatFn,
    ) -> tuple[str, list[dict[str, str]], dict[str, Any], list[str]]:
        """Return (assistant_text, updated_messages, updated_state, used_tools)."""
        working_messages = list(messages)
        safe_state = dict(state or {})
        used_tools: list[str] = []
        last_tool_result: tuple[str, dict[str, Any]] | None = None
        latest_user_text = self._latest_user_text(working_messages)

        for round_idx in range(self.config.max_tool_rounds + 1):
            request_messages = self._compact_messages(working_messages)
            force_final = round_idx >= self.config.max_tool_rounds
            compact_mode = len(request_messages) >= 5
            controller_prompt = self._build_controller_prompt(
                force_final=force_final,
                hint=self._tool_hint_for_user(latest_user_text) if round_idx == 0 else "",
                compact_mode=compact_mode,
            )
            draft = self._chat_with_retry(
                llm_chat=llm_chat,
                request_messages=request_messages,
                prompt=controller_prompt,
            )

            decision = self._parse_controller_output(draft)
            if decision and decision["action"] == "tool" and not force_final:
                tool_name = self._normalize_tool_name(decision["tool_name"])
                if not tool_name:
                    tool_name = decision["tool_name"]
                tool_result, safe_state = self.tool_system.execute(
                    tool_name,
                    decision["arguments"],
                    phone=phone,
                    state=safe_state,
                )
                last_tool_result = (tool_name, tool_result)
                used_tools.append(tool_name)
                working_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "[TOOL_CALL] "
                            + json.dumps(
                                {
                                    "tool_name": tool_name,
                                    "arguments": decision["arguments"],
                                },
                                ensure_ascii=False,
                            )
                        ),
                    }
                )
                working_messages.append(
                    {
                        "role": "system",
                        "content": "[TOOL_RESULT] "
                        + self._tool_result_to_text(tool_name, tool_result),
                    }
                )
                continue

            if decision and decision["action"] == "answer":
                answer = self._clean_answer_text(decision.get("answer", ""))
            else:
                answer = self._clean_answer_text(draft)
            if answer:
                answer = self._repair_answer_if_needed(answer, last_tool_result)
                working_messages.append({"role": "assistant", "content": answer})
                return answer, self._compact_messages(working_messages), safe_state, used_tools

        final_prompt = (
            "Bạn đã có dữ liệu [TOOL_RESULT] nếu đã gọi tool. "
            "Hãy trả lời trực tiếp cho khách hàng bằng tiếng Việt có dấu, ngắn gọn, đúng dữ liệu. "
            "Tối đa 4 câu. Nếu có danh sách giá gói, chỉ nêu tên gói và giá/tháng, không thêm giải thích dài."
        )
        final_answer = self._chat_with_retry(
            llm_chat=llm_chat,
            request_messages=self._compact_messages(working_messages),
            prompt=final_prompt,
        )
        final_answer = self._clean_answer_text(final_answer)
        final_answer = self._repair_answer_if_needed(final_answer, last_tool_result)
        if not final_answer:
            final_answer = "Xin lỗi, tôi chưa nghe rõ. Bạn có thể nói lại giúp tôi không?"
        working_messages.append({"role": "assistant", "content": final_answer})
        return final_answer, self._compact_messages(working_messages), safe_state, used_tools

    def _build_controller_prompt(
        self,
        force_final: bool,
        hint: str = "",
        compact_mode: bool = False,
    ) -> str:
        tool_brief = self._build_tool_brief()
        short_tools = ", ".join(item["name"] for item in self.tool_system.get_tool_schemas())
        if force_final:
            return (
                "Bạn là controller. Chỉ trả về đúng 1 JSON object duy nhất theo dạng: "
                '{"action":"answer","answer":"..."}. '
                "Không gọi tool. Nếu có kết quả tool trước đó thì phải dùng kết quả đó."
            )

        if compact_mode:
            base = (
                "Chỉ trả JSON duy nhất: "
                '{"action":"tool","tool_name":"...","arguments":{}} '
                "hoặc "
                '{"action":"answer","answer":"..."}. '
                "Câu hỏi giá/bao nhiêu/danh sách/tra cứu thuê bao phải dùng tool. "
                "Tools: "
                + short_tools
                + "."
            )
            if hint:
                base += " Gợi ý: " + hint
            return base

        base = (
            "Bạn là controller cho trợ lý VNPost Telecom.\n"
            "Chỉ trả về đúng 1 JSON object duy nhất, không thêm chữ nào khác.\n"
            "Có 2 dạng hợp lệ:\n"
            '1) {"action":"tool","tool_name":"<name>","arguments":{...}}\n'
            '2) {"action":"answer","answer":"<noi_dung_tieng_viet_co_dau>"}\n'
            "Quy tắc: nếu câu hỏi cần số liệu cụ thể (giá, bao nhiêu, danh sách gói, tra cứu thuê bao) thì bắt buộc action=tool.\n"
            "Ví dụ: 'bao nhiêu dịch vụ' -> count_mobile_services; 'giá các gói cước' -> list_mobile_plans với focus='price'; "
            "'dịch vụ viễn thông di động' -> list_mobile_service_groups.\n"
            "Danh sách tools: "
            + tool_brief
            + "\nKhông dùng markdown. Không dùng code fence."
        )
        if hint:
            base += "\nGợi ý điều phối: " + hint
        return base

    def _build_tool_brief(self) -> str:
        chunks: list[str] = []
        for schema in self.tool_system.get_tool_schemas():
            name = str(schema.get("name", "")).strip()
            desc = str(schema.get("description", "")).strip()
            arg_props = schema.get("arguments", {}).get("properties", {})
            arg_names = list(arg_props.keys()) if isinstance(arg_props, dict) else []
            if arg_names:
                chunks.append(f"- {name}({', '.join(arg_names)}): {desc}")
            else:
                chunks.append(f"- {name}(): {desc}")
        return "\n".join(chunks)

    def _normalize_tool_name(self, raw_name: str) -> str | None:
        candidate = raw_name.strip()
        if not candidate:
            return None

        alias = {
            "list_mobile_services": "list_mobile_service_groups",
            "count_services": "count_mobile_services",
            "list_plans": "list_mobile_plans",
            "recommend_plans": "recommend_mobile_plans",
            "list_services": "list_mobile_service_groups",
        }
        if candidate in alias:
            return alias[candidate]

        known = [item["name"] for item in self.tool_system.get_tool_schemas()]
        if candidate in known:
            return candidate

        stripped = re.sub(r"[^a-zA-Z0-9_]", "", candidate)
        if stripped in alias:
            return alias[stripped]
        for name in known:
            if stripped == name:
                return name
        for name in known:
            if name in candidate:
                return name
        return None

    def _latest_user_text(self, messages: list[dict[str, str]]) -> str:
        for item in reversed(messages):
            if item.get("role") == "user":
                return str(item.get("content", ""))
        return ""

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower().replace("đ", "d")
        nfd = unicodedata.normalize("NFD", lowered)
        return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")

    def _tool_hint_for_user(self, user_text: str) -> str:
        norm = self._normalize_text(user_text)
        if not norm:
            return ""

        if ("bao nhieu" in norm or "so luong" in norm or "tong" in norm) and (
            "dich vu" in norm or "vien thong" in norm
        ):
            return (
                "Câu hỏi dạng đếm số lượng dịch vụ, ưu tiên tool "
                '{"action":"tool","tool_name":"count_mobile_services","arguments":{}}.'
            )

        if ("gia" in norm or "bang gia" in norm or "bao nhieu tien" in norm) and "goi" in norm:
            return (
                "Câu hỏi báo giá gói cước, ưu tiên tool "
                '{"action":"tool","tool_name":"list_mobile_plans","arguments":{"focus":"price"}}.'
            )

        if "dich vu vien thong di dong" in norm or "vien thong di dong" in norm:
            return (
                "Câu hỏi về nhóm dịch vụ di động, ưu tiên tool "
                '{"action":"tool","tool_name":"list_mobile_service_groups","arguments":{}}.'
            )

        if "tra cuu" in norm and "thue bao" in norm:
            return (
                "Câu hỏi tra cứu thuê bao, ưu tiên tool "
                '{"action":"tool","tool_name":"lookup_subscriber","arguments":{}}.'
            )
        return ""

    def _compact_messages(
        self,
        messages: list[dict[str, str]],
        max_items: int = 6,
    ) -> list[dict[str, str]]:
        if len(messages) <= max_items:
            return [self._truncate_message(item) for item in messages]

        first = messages[0] if messages and messages[0].get("role") == "system" else None
        tail = list(messages[1:] if first else messages)

        tool_call_indices = [
            idx
            for idx, item in enumerate(tail)
            if item.get("role") == "assistant"
            and str(item.get("content", "")).startswith("[TOOL_CALL]")
        ]
        tool_result_indices = [
            idx
            for idx, item in enumerate(tail)
            if item.get("role") == "system"
            and str(item.get("content", "")).startswith("[TOOL_RESULT]")
        ]
        last_tool_call = tool_call_indices[-1] if tool_call_indices else -1
        last_tool_result = tool_result_indices[-1] if tool_result_indices else -1

        filtered_tail: list[dict[str, str]] = []
        for idx, item in enumerate(tail):
            content = str(item.get("content", ""))
            role = str(item.get("role", ""))
            if role == "assistant" and content.startswith("[TOOL_CALL]") and idx != last_tool_call:
                continue
            if role == "system" and content.startswith("[TOOL_RESULT]") and idx != last_tool_result:
                continue
            filtered_tail.append(item)

        tail_limit = max(1, max_items - 1 if first else max_items)
        compact_tail = filtered_tail[-tail_limit:]
        compact_tail = [self._truncate_message(item) for item in compact_tail]
        if first is None:
            return compact_tail
        return [self._truncate_message(first), *compact_tail]

    def _truncate_message(
        self,
        message: dict[str, str],
        max_chars: int = 320,
    ) -> dict[str, str]:
        content = str(message.get("content", ""))
        if len(content) <= max_chars:
            return dict(message)
        return {"role": str(message.get("role", "user")), "content": content[:max_chars] + "..."}

    def _parse_controller_output(self, text: str) -> dict[str, Any] | None:
        payload = self._extract_json_object(text)
        if isinstance(payload, dict):
            decision = self._parse_payload_dict(payload)
            if decision:
                return decision

        return self._parse_jsonish_text(text)

    def _chat_with_retry(
        self,
        llm_chat: ChatFn,
        request_messages: list[dict[str, str]],
        prompt: str,
    ) -> str:
        payload = [*request_messages, {"role": "system", "content": prompt}]
        try:
            return llm_chat(payload).strip()
        except Exception as exc:
            err = str(exc).lower()
            if not any(
                key in err
                for key in (
                    "badrequest",
                    "context length",
                    "maximum input length",
                    "input tokens",
                    "too many tokens",
                )
            ):
                raise

            fallback_messages = self._compact_messages(request_messages, max_items=4)
            fallback_payload = [*fallback_messages, {"role": "system", "content": prompt}]
            return llm_chat(fallback_payload).strip()

    def _parse_payload_dict(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        action_raw = str(payload.get("action", "")).strip()
        action = action_raw.lower()
        tool_name_raw = str(payload.get("tool_name", payload.get("name", ""))).strip()
        args = payload.get("arguments", payload.get("args", {}))
        if not isinstance(args, dict):
            args = {}

        action_as_tool = self._normalize_tool_name(action_raw)
        if action_as_tool:
            return {"action": "tool", "tool_name": action_as_tool, "arguments": args}

        if action in {"tool", "call_tool"} or (tool_name_raw and action != "answer"):
            tool_name = self._normalize_tool_name(tool_name_raw)
            if tool_name:
                return {"action": "tool", "tool_name": tool_name, "arguments": args}
            return None

        if action == "answer" or "answer" in payload:
            answer = str(payload.get("answer", "")).strip()
            return {"action": "answer", "answer": answer}

        return None

    def _parse_jsonish_text(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None

        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', stripped)
        action_raw = action_match.group(1).strip() if action_match else ""
        action = action_raw.lower()

        if action == "answer":
            return {"action": "answer", "answer": self._extract_answer_from_text(stripped)}

        action_as_tool = self._normalize_tool_name(action_raw)
        if action_as_tool:
            return {
                "action": "tool",
                "tool_name": action_as_tool,
                "arguments": self._extract_arguments_from_text(stripped),
            }

        if action in {"tool", "call_tool"}:
            tool_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', stripped)
            if not tool_match:
                tool_match = re.search(r'"name"\s*:\s*"([^"]+)"', stripped)
            tool_raw = tool_match.group(1).strip() if tool_match else ""
            tool_name = self._normalize_tool_name(tool_raw)
            if tool_name:
                return {
                    "action": "tool",
                    "tool_name": tool_name,
                    "arguments": self._extract_arguments_from_text(stripped),
                }
            return None

        tool_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', stripped)
        if tool_match:
            tool_name = self._normalize_tool_name(tool_match.group(1).strip())
            if tool_name:
                return {
                    "action": "tool",
                    "tool_name": tool_name,
                    "arguments": self._extract_arguments_from_text(stripped),
                }

        if '"answer"' in stripped and stripped.startswith("{"):
            return {"action": "answer", "answer": self._extract_answer_from_text(stripped)}
        return None

    def _extract_arguments_from_text(self, text: str) -> dict[str, Any]:
        for key in ('"arguments"', '"args"'):
            idx = text.find(key)
            if idx < 0:
                continue
            colon = text.find(":", idx + len(key))
            if colon < 0:
                continue
            brace_start = text.find("{", colon + 1)
            if brace_start < 0:
                continue
            raw_obj = self._extract_balanced_braces(text, brace_start)
            if not raw_obj:
                continue
            try:
                obj = json.loads(raw_obj)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return {}

    def _extract_balanced_braces(self, text: str, start: int) -> str | None:
        if start < 0 or start >= len(text) or text[start] != "{":
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _extract_answer_from_text(self, text: str) -> str:
        key_match = re.search(r'"answer"\s*:\s*', text)
        if not key_match:
            return ""

        start = key_match.end()
        if start >= len(text):
            return ""

        if text[start] == '"':
            chars: list[str] = []
            escaped = False
            for ch in text[start + 1 :]:
                if escaped:
                    chars.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    break
                chars.append(ch)
            raw = "".join(chars)
            return raw.replace("\\n", "\n").strip()

        tail = text[start:]
        tail = re.split(r"[,}]", tail, maxsplit=1)[0]
        return tail.strip().strip('"').strip()

    def _tool_result_to_text(self, tool_name: str, payload: dict[str, Any]) -> str:
        if not payload.get("ok"):
            return f"{tool_name}: ERROR {payload.get('error', 'unknown')}"

        result = payload.get("result", {})
        if not isinstance(result, dict):
            return f"{tool_name}: {str(result)}"

        if tool_name == "count_mobile_services":
            groups = result.get("service_groups", [])
            group_map = {
                str(item.get("group", "")): int(item.get("count", 0))
                for item in groups
                if isinstance(item, dict)
            }
            return (
                f"count_mobile_services: mobile_plans={group_map.get('mobile_plans', 0)}, "
                f"sim_offers={group_map.get('sim_offers', 0)}, "
                f"total_products={result.get('total_products', 0)}"
            )

        if tool_name == "list_mobile_service_groups":
            groups = result.get("groups", [])
            if not isinstance(groups, list):
                groups = []
            parts = []
            for item in groups[:4]:
                if not isinstance(item, dict):
                    continue
                parts.append(
                    f"{item.get('group', '')}: {item.get('label_vi', '')} - {item.get('description_vi', '')}"
                )
            return "list_mobile_service_groups: " + " | ".join(parts)

        if tool_name == "list_mobile_plans":
            plans = result.get("plans", [])
            if not isinstance(plans, list):
                plans = []
            parts = []
            for plan in plans[:6]:
                if not isinstance(plan, dict):
                    continue
                parts.append(
                    f"{plan.get('id', '')}/{plan.get('name', '')}: "
                    f"{plan.get('monthly_price_vnd', 0)} VND, "
                    f"{plan.get('data_gb_per_day', 0)}GB/day, "
                    f"{plan.get('call_minutes_onnet', 0)} min onnet, "
                    f"5g={plan.get('is_5g', False)}"
                )
            return (
                "list_mobile_plans: total="
                + str(result.get("total", 0))
                + "; "
                + " | ".join(parts)
            )

        if tool_name == "recommend_mobile_plans":
            recs = result.get("recommendations", [])
            if not isinstance(recs, list):
                recs = []
            parts = []
            for rec in recs[:4]:
                if not isinstance(rec, dict):
                    continue
                plan = rec.get("plan", {}) if isinstance(rec.get("plan"), dict) else {}
                parts.append(
                    f"{rec.get('plan_id', '')}: {plan.get('monthly_price_vnd', 0)} VND, "
                    f"{plan.get('data_gb_per_day', 0)}GB/day, reason={rec.get('reason', '')}"
                )
            return "recommend_mobile_plans: " + " | ".join(parts)

        if tool_name == "list_sim_offers":
            offers = result.get("offers", [])
            if not isinstance(offers, list):
                offers = []
            parts = []
            for offer in offers[:5]:
                if not isinstance(offer, dict):
                    continue
                parts.append(
                    f"{offer.get('id', '')}/{offer.get('name', '')}: "
                    f"{offer.get('sim_price_vnd', 0)} VND, {offer.get('bundle', '')}"
                )
            return "list_sim_offers: " + " | ".join(parts)

        if tool_name == "lookup_subscriber":
            if not result.get("found"):
                return f"lookup_subscriber: not found phone={result.get('phone', '')}"
            subscriber = result.get("subscriber", {}) if isinstance(result.get("subscriber"), dict) else {}
            plan = result.get("current_plan", {}) if isinstance(result.get("current_plan"), dict) else {}
            return (
                f"lookup_subscriber: {subscriber.get('phone', '')} {subscriber.get('name', '')}; "
                f"plan={plan.get('id', '')}/{plan.get('name', '')}; "
                f"price={plan.get('monthly_price_vnd', 0)}; "
                f"data_usage={subscriber.get('avg_data_usage_gb_per_day', 0)}GB/day; "
                f"call_usage={subscriber.get('avg_call_minutes_per_month', 0)}min/month"
            )

        if tool_name == "create_lead":
            return (
                f"create_lead: lead_id={result.get('lead_id', '')}, "
                f"status={result.get('status', '')}, offer={result.get('offer', {})}"
            )

        if tool_name == "create_callback":
            return (
                f"create_callback: callback_id={result.get('callback_id', '')}, "
                f"status={result.get('status', '')}, after_hours={result.get('scheduled_after_hours', '')}"
            )

        raw = json.dumps(payload, ensure_ascii=False)
        return raw[:500] + ("..." if len(raw) > 500 else "")

    def _extract_json_object(self, text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            return None

        candidates = [stripped]

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
        if fence_match:
            candidates.insert(0, fence_match.group(1).strip())

        brace_match = re.search(r"(\{.*\})", stripped, flags=re.S)
        if brace_match:
            candidates.append(brace_match.group(1).strip())

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _clean_answer_text(self, text: str) -> str:
        cleaned = text.strip()
        parsed = self._parse_jsonish_text(cleaned)
        if parsed:
            if parsed["action"] == "answer":
                return str(parsed.get("answer", "")).strip()
            # Never surface control JSON to user.
            return ""

        json_answer_match = re.search(
            r'"action"\s*:\s*"answer".*?"answer"\s*:\s*"(.*)"\s*}\s*$',
            cleaned,
            flags=re.S,
        )
        if cleaned.startswith("{") and json_answer_match:
            answer = json_answer_match.group(1)
            answer = answer.replace('\\"', '"').replace("\\n", "\n")
            return answer.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        if cleaned.startswith("[TOOL_"):
            return ""
        if cleaned.startswith("{") and '"action"' in cleaned:
            return ""
        return self._trim_incomplete_tail(cleaned)

    def _trim_incomplete_tail(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return stripped
        if stripped[-1] in ".!?":
            return stripped

        if stripped.count("**") % 2 == 1:
            stripped = stripped.rsplit("**", 1)[0].rstrip()

        last_cut = max(
            stripped.rfind("."),
            stripped.rfind("!"),
            stripped.rfind("?"),
            stripped.rfind(";"),
            stripped.rfind("\n"),
            stripped.rfind(","),
        )
        if last_cut >= max(20, len(stripped) // 2):
            trimmed = stripped[: last_cut + 1].strip()
            if trimmed and trimmed[-1] == ",":
                trimmed = trimmed[:-1].rstrip() + "."
            if trimmed:
                return trimmed
        return stripped

    def _repair_answer_if_needed(
        self,
        answer: str,
        last_tool_result: tuple[str, dict[str, Any]] | None,
    ) -> str:
        fixed = answer.strip()
        if not last_tool_result:
            return fixed

        tool_name, payload = last_tool_result

        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not isinstance(result, dict):
            return fixed

        if tool_name == "list_mobile_plans":
            plans = result.get("plans", [])
            if isinstance(plans, list) and plans:
                currency_count = len(re.findall(r"\b(?:VND|VNĐ)\b", fixed, flags=re.I))
                if self._is_likely_truncated(fixed) or currency_count < max(1, min(len(plans), 4)):
                    return self._format_plan_price_answer(plans)
            return fixed

        if not self._is_likely_truncated(fixed):
            return fixed

        if tool_name == "recommend_mobile_plans":
            recs = result.get("recommendations", [])
            if isinstance(recs, list) and recs:
                top = recs[0] if isinstance(recs[0], dict) else {}
                plan = top.get("plan", {}) if isinstance(top.get("plan"), dict) else {}
                if plan:
                    price = int(plan.get("monthly_price_vnd", 0))
                    return (
                        f"Gói phù hợp: {plan.get('name', top.get('plan_id', ''))} "
                        f"({price:,} VND/tháng), "
                        f"{plan.get('data_gb_per_day', 0)}GB/ngày."
                    ).replace(",", ".")
        return fixed

    def _is_likely_truncated(self, text: str) -> bool:
        s = text.strip()
        if not s:
            return True
        if len(s) < 40:
            return False
        if s[-1] in ".!?":
            return False
        if s[-1] in {":", "-", ";", ",", "(", "["}:
            return True
        if re.search(r"\b\d{1,3},\d{1,2}$", s):
            return True
        if re.search(r"\b\d+\.$", s):
            return True
        if re.search(r"\bVP_[A-Z0-9_]{1,6}$", s):
            return True
        if "\n- " in s:
            return True
        return False

    def _format_plan_price_answer(self, plans: list[dict[str, Any]]) -> str:
        rows: list[str] = []
        for plan in plans[:4]:
            if not isinstance(plan, dict):
                continue
            price = int(plan.get("monthly_price_vnd", 0))
            rows.append(
                f"{plan.get('name', plan.get('id', ''))}: {price:,} VND/tháng".replace(",", ".")
            )
        if not rows:
            return "Em chưa có dữ liệu giá gói cước phù hợp."
        return "Giá các gói cước: " + "; ".join(rows) + "."
