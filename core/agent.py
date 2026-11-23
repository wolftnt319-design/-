"""Stage 5: LUNA Agent Core — Verified Execution, Memory RAG, Confirmation, & Teacher Co-Pilot."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional

from config import config_manager
from core.context_manager import context_manager
from core.language import detect_language
from core.model_manager import model_manager
from core.online_teacher import online_teacher
from core.permissions import RiskLevel, audit_logger, confirmation_manager
from core.response_validator import ResponseValidator
from core.router import HybridRouter, RouteDecision
from memory.long_memory import long_memory
from memory.retrieval import retriever
from memory.short_memory import short_memory
from tools.registry import ToolRegistry

YES_CONFIRM_RE = re.compile(
    r"^(?:بله|آره|تایید|تأیید|ذخیره\s+کن|باشه|انجام\s+بده|موافقم|yes|confirm|approve|ok|sure)\s*[.!؟]*$",
    re.I,
)
NO_REJECT_RE = re.compile(
    r"^(?:نه|خیر|لغو|انصراف|ذخیره\s+نکن|انجام\s+نده|no|cancel|reject)\s*[.!؟]*$",
    re.I,
)

EventCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class LunaAgent:
    """End-to-end local-first AI Agent orchestrating Routing, Tools, Memory, Security, and Teacher Distillation."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or ToolRegistry()
        self.router = HybridRouter(self.registry)
        self._cancel_flags: Dict[str, bool] = {}

    def cancel_session(self, session_id: str) -> None:
        self._cancel_flags[session_id] = True
        audit_logger.log("agent_operation_cancelled", "user", "cancelled", session_id=session_id)

    async def _emit(self, callback: Optional[EventCallback], payload: Dict[str, Any]) -> None:
        if callback is not None:
            try:
                await callback(payload)
            except Exception:
                pass

    async def process_message(
        self,
        user_message: str,
        session_id: str = "default",
        event_callback: Optional[EventCallback] = None,
    ) -> Dict[str, Any]:
        """Main agent turn with step cap, wall-clock timeout, confirmation broker, and provenance tracking."""
        t_start = time.perf_counter()
        self._cancel_flags[session_id] = False
        settings = config_manager.settings

        short_ctx = short_memory.get_context_window(session_id)
        session_pref = short_ctx.get("language", "auto")
        pref = session_pref if session_pref in ("fa", "en") else settings.preferred_language
        language, explicit_switch = detect_language(user_message, preferred=pref)  # type: ignore[arg-type]

        if explicit_switch is not None:
            short_memory.set_session_language(session_id, explicit_switch)
            language = explicit_switch

        # 0. Check if user replied "بله / خیر" in chat to resolve a pending confirmation in this session!
        pending_list = confirmation_manager.list_pending(session_id=session_id)
        if pending_list and (YES_CONFIRM_RE.match(user_message.strip()) or NO_REJECT_RE.match(user_message.strip())):
            approved = bool(YES_CONFIRM_RE.match(user_message.strip()))
            short_memory.add_message(session_id, "user", user_message, provenance="user")
            latest_token = pending_list[-1]["token"]
            return await self.resolve_confirmation(latest_token, approved=approved, language=language)

        short_memory.add_message(session_id, "user", user_message, provenance="user")

        # Handle pure language-switch command (`/lang en` or `"انگلیسی صحبت کن"` with no other task)
        if user_message.strip().lower() in ("/lang en", "/lang fa"):
            reply = "زبان گفتگو به فارسی تغییر یافت." if language == "fa" else "Conversation language switched to English."
            short_memory.add_message(session_id, "assistant", reply, provenance="system")
            return {
                "ok": True,
                "session_id": session_id,
                "language": language,
                "reply": reply,
                "provenance": "system",
                "route": {"intent": "chat", "action": "language_switch"},
                "executed_tools": [],
                "memories_used": [],
                "pending_confirmation": None,
                "latency_ms": int((time.perf_counter() - t_start) * 1000),
            }

        # 1. Route User Intent
        await self._emit(event_callback, {"type": "status", "stage": "routing", "message": "در حال تحلیل درخواست..."})
        decision = await self.router.route(user_message, session_id=session_id)

        # Blocked unsafe shell commands
        if decision.action == "blocked_unsafe_shell":
            reply = (
                "به دلایل امنیتی، اجرای مستقیم فرمان‌های سیستم/شِل یا عملیات مخرب در لونا مسدود است. "
                "فقط ابزارهای مجاز داخل فضای کاری قابل استفاده هستند."
                if language == "fa"
                else "Direct shell or destructive OS commands are blocked by LUNA's security policy."
            )
            short_memory.add_message(session_id, "assistant", reply, provenance="security")
            return {
                "ok": True,
                "session_id": session_id,
                "language": language,
                "reply": reply,
                "provenance": "security",
                "route": decision.to_dict(),
                "executed_tools": [],
                "memories_used": [],
                "pending_confirmation": None,
                "latency_ms": int((time.perf_counter() - t_start) * 1000),
            }

        # 2. Retrieve Relevant Long-Term Memories & Distilled Teacher Exemplars
        memories = retriever.search_memories(user_message, top_k=settings.rag_top_k)
        exemplars = retriever.search_exemplars(user_message, language=language, top_k=2)

        # If user explicitly asked a memory recall question and NO memory exists -> state explicitly!
        if decision.intent == "memory" and decision.action == "memory_search" and not memories:
            reply = (
                "اطلاعات ذخیره‌شده‌ای در حافظه بلندمدت درباره این موضوع وجود ندارد. "
                "اگر مایلید نکته‌ای را به خاطر بسپارم، آن را بیان کنید."
                if language == "fa"
                else "There is no stored information in long-term memory about this topic yet."
            )
            short_memory.add_message(session_id, "assistant", reply, provenance="memory")
            return {
                "ok": True,
                "session_id": session_id,
                "language": language,
                "reply": reply,
                "provenance": "memory",
                "route": decision.to_dict(),
                "executed_tools": [],
                "memories_used": [],
                "pending_confirmation": None,
                "latency_ms": int((time.perf_counter() - t_start) * 1000),
            }

        # 3. Handle Multi-Step Notepad Story Flow
        if decision.action == "notepad_story_flow":
            return await self._execute_notepad_story_flow(
                user_message=user_message,
                decision=decision,
                language=language,
                session_id=session_id,
                short_ctx=short_ctx,
                exemplars=exemplars,
                event_callback=event_callback,
                t_start=t_start,
            )

        # 4. Confirmation Check for Sensitive / Interactive Memory Actions
        if decision.action not in ("chat_reply", "none") and decision.requires_confirmation:
            summary_fa, summary_en = self._describe_confirmation(decision)
            pending = confirmation_manager.request_confirmation(
                action_type="memory" if decision.intent == "memory" else "tool",
                tool_name=decision.action,
                arguments=decision.arguments,
                summary_fa=summary_fa,
                summary_en=summary_en,
                risk_level=int(RiskLevel.L3_SENSITIVE_MUTATE),
                session_id=session_id,
            )
            reply = (
                f"متوجه شدم. {summary_fa}\n(می‌توانید دکمه «تأیید» را بزنید یا عبارت «بله» را ارسال کنید.)"
                if language == "fa"
                else f"Understood. {summary_en}\n(Click Approve or reply 'yes' to proceed.)"
            )
            short_memory.add_message(
                session_id,
                "assistant",
                reply,
                provenance="confirmation",
                meta={"pending_confirmation": pending.to_public_dict()},
            )
            return {
                "ok": True,
                "session_id": session_id,
                "language": language,
                "reply": reply,
                "provenance": "confirmation",
                "route": decision.to_dict(),
                "executed_tools": [],
                "memories_used": memories,
                "pending_confirmation": pending.to_public_dict(),
                "latency_ms": int((time.perf_counter() - t_start) * 1000),
            }

        # 5. Execute Bounded Multi-Step Tool Plan (if routed to tool)
        executed_tools: List[Dict[str, Any]] = []
        web_observations: List[str] = []
        pending_web_memory: Optional[Dict[str, Any]] = None

        steps_queue: List[RouteDecision] = []
        if decision.action not in ("chat_reply", "none", "memory_search"):
            steps_queue.append(decision)
            steps_queue.extend(decision.secondary_steps[: settings.security.max_agent_steps - 1])

        seen_calls: set[str] = set()
        for step_idx, step in enumerate(steps_queue):
            if step_idx >= settings.security.max_agent_steps:
                break
            if self._cancel_flags.get(session_id):
                break
            if (time.perf_counter() - t_start) > settings.security.agent_wall_timeout_seconds:
                break

            call_sig = f"{step.action}:{sorted(step.arguments.items())}"
            if call_sig in seen_calls:
                break  # 2-strike cycle breaker
            seen_calls.add(call_sig)

            # Substitute placeholder from previous step if multi-step
            resolved_args = dict(step.arguments)
            if executed_tools and "{{LAST_TOOL_SUMMARY}}" in str(resolved_args):
                last_sum = str(executed_tools[-1].get("summary") or executed_tools[-1].get("result") or "")
                for k, v in resolved_args.items():
                    if isinstance(v, str) and "{{LAST_TOOL_SUMMARY}}" in v:
                        resolved_args[k] = v.replace("{{LAST_TOOL_SUMMARY}}", last_sum)

            await self._emit(
                event_callback,
                {"type": "tool_start", "tool": step.action, "arguments": resolved_args},
            )
            tool_res = await self.registry.execute(step.action, resolved_args, session_id=session_id)
            executed_tools.append(tool_res)
            await self._emit(event_callback, {"type": "tool_end", "tool": step.action, "result": tool_res})

            # Stop downstream steps immediately if a prerequisite tool failed!
            if not tool_res.get("ok"):
                if tool_res.get("requires_confirmation"):
                    pending = confirmation_manager.request_confirmation(
                        action_type="tool",
                        tool_name=step.action,
                        arguments=resolved_args,
                        summary_fa=f"آیا اجرای عملیات حساس «{step.action}» را تأیید می‌کنید؟",
                        summary_en=f"Do you approve running sensitive action '{step.action}'?",
                        session_id=session_id,
                    )
                    reply = tool_res.get("error", "نیاز به تأیید کاربر دارد.")
                    short_memory.add_message(session_id, "assistant", reply, provenance="confirmation")
                    return {
                        "ok": True,
                        "session_id": session_id,
                        "language": language,
                        "reply": reply,
                        "provenance": "confirmation",
                        "route": decision.to_dict(),
                        "executed_tools": executed_tools,
                        "memories_used": memories,
                        "pending_confirmation": pending.to_public_dict(),
                        "latency_ms": int((time.perf_counter() - t_start) * 1000),
                    }
                break

            if step.intent in ("search", "browser"):
                if tool_res.get("untrusted_content"):
                    web_observations.append(tool_res["untrusted_content"])
                elif tool_res.get("results"):
                    joined_snips = "\n".join(r.get("untrusted_wrapped", "") for r in tool_res["results"][:3])
                    web_observations.append(joined_snips)
                if tool_res.get("suggested_memory_note"):
                    note = tool_res["suggested_memory_note"]
                    pending_mem = confirmation_manager.request_confirmation(
                        action_type="memory",
                        tool_name="memory_save",
                        arguments={
                            "content": note["content"],
                            "title": note["title"],
                            "category": "web_note",
                            "importance": 3,
                        },
                        summary_fa=f"آیا خلاصه این یافته وب در حافظه بلندمدت ذخیره شود؟ ({note['title']})",
                        summary_en=f"Save this web insight to long-term memory? ({note['title']})",
                        risk_level=int(RiskLevel.L1_WORKSPACE_WRITE),
                        session_id=session_id,
                    )
                    pending_web_memory = pending_mem.to_public_dict()

        # 6. Synthesize Final Grounded Answer & Validate Honesty
        reply, provenance, arena_comparison = await self._synthesize_final_answer(
            user_message=user_message,
            language=language,
            short_ctx=short_ctx,
            memories=memories,
            exemplars=exemplars,
            executed_tools=executed_tools,
            web_observations=web_observations,
            intent=decision.intent,
            event_callback=event_callback,
        )

        short_memory.add_message(
            session_id,
            "assistant",
            reply,
            provenance=provenance,
            meta={
                "route": decision.to_dict(),
                "tools": [{"tool": t.get("tool"), "ok": t.get("ok")} for t in executed_tools],
                "arena_comparison": arena_comparison,
            },
        )

        return {
            "ok": True,
            "session_id": session_id,
            "language": language,
            "reply": reply,
            "provenance": provenance,
            "route": decision.to_dict(),
            "executed_tools": executed_tools,
            "memories_used": memories,
            "exemplars_used": len(exemplars),
            "pending_confirmation": pending_web_memory,
            "arena_comparison": arena_comparison,
            "latency_ms": int((time.perf_counter() - t_start) * 1000),
        }

    async def resolve_confirmation(
        self,
        token: str,
        approved: bool,
        language: Literal["fa", "en"] = "fa",
    ) -> Dict[str, Any]:
        item = confirmation_manager.pop_decision(token, approved=approved)
        if item is None:
            msg = (
                "درخواست تأیید منقضی شده یا یافت نشد."
                if language == "fa"
                else "Confirmation token expired or not found."
            )
            return {"ok": False, "reply": msg, "provenance": "security"}

        if not approved:
            msg = (
                f"عملیات «{item.tool_name}» به درخواست شما لغو شد و هیچ تغییری اعمال نگردید."
                if language == "fa"
                else f"Operation '{item.tool_name}' was cancelled at your request."
            )
            short_memory.add_message(item.session_id, "assistant", msg, provenance="security")
            return {"ok": True, "approved": False, "reply": msg, "provenance": "security"}

        # Execute the now-approved action
        tool_res = await self.registry.execute(item.tool_name, item.arguments, session_id=item.session_id)
        if tool_res.get("ok"):
            if item.tool_name == "memory_save":
                action_kind = tool_res.get("action", "created")
                msg = (
                    f"در حافظه بلندمدت ذخیره شد ({'ادغام با مورد مشابه' if 'merged' in action_kind else 'ثبت جدید'}): «{item.arguments.get('content', '')[:90]}»"
                    if language == "fa"
                    else f"Saved to long-term memory ({action_kind})."
                )
                prov = "memory"
            else:
                msg = tool_res.get("summary") or "عملیات تأییدشده با موفقیت انجام شد."
                prov = "tool"
        else:
            msg = (
                f"اجرای عملیات پس از تأیید با خطا مواجه شد: {tool_res.get('error')}"
                if language == "fa"
                else f"Approved operation failed: {tool_res.get('error')}"
            )
            prov = "tool"

        short_memory.add_message(item.session_id, "assistant", msg, provenance=prov, meta={"tool_result": tool_res})
        return {
            "ok": bool(tool_res.get("ok")),
            "approved": True,
            "reply": msg,
            "provenance": prov,
            "executed_tools": [tool_res],
        }

    async def _execute_notepad_story_flow(
        self,
        user_message: str,
        decision: RouteDecision,
        language: Literal["fa", "en"],
        session_id: str,
        short_ctx: Dict[str, Any],
        exemplars: List[Dict[str, Any]],
        event_callback: Optional[EventCallback],
        t_start: float,
    ) -> Dict[str, Any]:
        """Stage 9: Generate story/note text first, validate it, then save & open securely via NotepadTool."""
        fname = decision.arguments.get("filename", "luna_story.txt")
        topic = decision.arguments.get("topic", user_message)

        await self._emit(event_callback, {"type": "status", "stage": "generating", "message": "در حال نگارش متن یادداشت/داستان..."})
        story_prompt = (
            f"یک متن یا داستان کوتاه، روان و خوش‌خوان به فارسی درباره موضوع زیر بنویس (حداکثر ۶ جمله):\n{topic}"
            if language == "fa"
            else f"Write a short, engaging text/story (up to 6 sentences) about:\n{topic}"
        )
        msgs = context_manager.build_messages(story_prompt, language=language, short_ctx=short_ctx, exemplars=exemplars)
        local_res = await model_manager.chat(msgs, language=language, timeout=35.0)
        val = ResponseValidator.validate_chat_response(local_res.content, story_prompt, language=language)

        story_text = val.cleaned_text
        prov = "tool"
        if (not local_res.ok or not val.valid) and config_manager.settings.operating_mode != "local_strict":
            t_res = await online_teacher.ask_teacher(story_prompt, language=language)
            if t_res.ok:
                story_text = t_res.content
                prov = "hybrid_teacher"

        if not story_text:
            story_text = f"یادداشت ثبت‌شده توسط LUNA AI:\nموضوع: {topic}"

        await self._emit(event_callback, {"type": "tool_start", "tool": "notepad_write", "arguments": {"filename": fname}})
        np_res = await self.registry.execute(
            "notepad_write",
            {"filename": fname, "content": story_text, "open_window": True},
            session_id=session_id,
        )
        await self._emit(event_callback, {"type": "tool_end", "tool": "notepad_write", "result": np_res})

        if np_res.get("ok"):
            reply = f"{np_res.get('summary')}\n\n---\n{story_text}"
        else:
            reply = f"خطا در ذخیره فایل یادداشت: {np_res.get('error')}"

        short_memory.add_message(session_id, "assistant", reply, provenance=prov)
        return {
            "ok": True,
            "session_id": session_id,
            "language": language,
            "reply": reply,
            "provenance": prov,
            "route": decision.to_dict(),
            "executed_tools": [np_res],
            "memories_used": [],
            "pending_confirmation": None,
            "latency_ms": int((time.perf_counter() - t_start) * 1000),
        }

    async def _synthesize_final_answer(
        self,
        user_message: str,
        language: Literal["fa", "en"],
        short_ctx: Dict[str, Any],
        memories: List[Dict[str, Any]],
        exemplars: List[Dict[str, Any]],
        executed_tools: List[Dict[str, Any]],
        web_observations: List[str],
        intent: str,
        event_callback: Optional[EventCallback],
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        settings = config_manager.settings

        # A) If any tool failed -> report exact failure honestly without letting a 0.6B LLM hallucinate success!
        if executed_tools and any(not t.get("ok") for t in executed_tools):
            failed_msgs = [
                f"ابزار `{t.get('tool')}` ناموفق بود: {t.get('error', 'خطای نامشخص')}"
                if language == "fa"
                else f"Tool `{t.get('tool')}` failed: {t.get('error', 'unknown error')}"
                for t in executed_tools
                if not t.get("ok")
            ]
            succeeded_msgs = [
                str(t.get("summary")) for t in executed_tools if t.get("ok") and t.get("summary")
            ]
            combined = "\n".join(succeeded_msgs + failed_msgs)
            return combined, ("internet" if intent in ("search", "browser") else "tool"), None

        # B) Deterministic exact tool results (Calculator, Word, Excel, File, Notepad) ->
        #    Always preserve exact verified numbers/paths/links!
        if executed_tools and intent in ("calculator", "word", "excel", "file", "notepad"):
            summaries = [str(t.get("summary", "")) for t in executed_tools if t.get("summary")]
            # Include read content preview when user read a file/docx/xlsx
            details_extra: List[str] = []
            for t in executed_tools:
                if t.get("result") is not None and intent == "calculator":
                    if language == "fa":
                        details_extra.append(f"پاسخ دقیق محاسبه: **{t['result']}** (`{t.get('expression')}`)")
                    else:
                        details_extra.append(f"Exact calculation result: **{t['result']}** (`{t.get('expression')}`)")
                elif t.get("content") and t.get("tool") == "file_read":
                    details_extra.append(f"\n```\n{str(t['content'])[:1200]}\n```")
                elif t.get("text") and t.get("tool") == "word_read":
                    details_extra.append(f"\n{str(t['text'])[:1200]}")
                elif t.get("rows") is not None and t.get("tool") == "excel_read":
                    preview_rows = "\n".join(" | ".join(str(c) for c in r) for r in t["rows"][:12])
                    if preview_rows:
                        details_extra.append(f"\n```\n{preview_rows}\n```")
                elif t.get("files") is not None and t.get("tool") == "file_list":
                    flist = "\n".join(f"- `{f['path']}` ({f['type']}, {f['size_bytes']} bytes)" for f in t["files"][:25])
                    if flist:
                        details_extra.append(f"\n{flist}")
            reply = "\n".join(details_extra if (intent == "calculator" and details_extra) else (summaries + details_extra)).strip()
            return reply, "tool", None

        # C) Web Search / Browser Results -> Summarize safely + ALWAYS attach verified links & access dates!
        if executed_tools and intent in ("search", "browser"):
            citations: List[str] = []
            snippets_for_synthesis: List[str] = []
            for t in executed_tools:
                if t.get("results"):
                    for idx, r in enumerate(t["results"][:4]):
                        citations.append(
                            f"- [{r['title']}]({r['url']}) — اعتبار: `{r['credibility']['tier']}` | دسترسی: `{r['accessed_at'][:19]}Z`"
                        )
                        snippets_for_synthesis.append(f"{idx+1}. {r['title']}: {r['snippet']}")
                elif t.get("url"):
                    citations.append(
                        f"- [{t.get('title') or t['url']}]({t['url']}) — اعتبار: `{t.get('credibility', {}).get('tier', 'medium')}` | دسترسی: `{t.get('accessed_at', '')[:19]}Z`"
                    )
                    snippets_for_synthesis.append(t.get("untrusted_content", "")[:1200])

            msgs = context_manager.build_messages(
                user_message=user_message,
                language=language,
                short_ctx=short_ctx,
                memories=memories,
                exemplars=exemplars,
                tool_observations=executed_tools,
                web_observations=web_observations,
            )
            local_res = await model_manager.chat(msgs, language=language, timeout=30.0)
            val = ResponseValidator.validate_chat_response(
                local_res.content, user_message, language=language, executed_tools=executed_tools
            )

            summary_body = val.cleaned_text if (local_res.ok and val.valid) else ""
            if not summary_body and settings.operating_mode != "local_strict" and settings.teacher.enabled:
                t_res = await online_teacher.ask_teacher(
                    user_message, language=language, context_snippets=snippets_for_synthesis
                )
                if t_res.ok:
                    summary_body = t_res.content

            if not summary_body:
                summary_body = "\n".join(snippets_for_synthesis[:3])

            sources_header = "منابع و تاریخ دسترسی:" if language == "fa" else "Sources & Access Timestamps:"
            final_web_reply = f"{summary_body}\n\n**{sources_header}**\n" + "\n".join(citations)
            return final_web_reply, "internet", None

        # D) Conversational / Memory-Grounded / Reasoning Turn
        await self._emit(event_callback, {"type": "status", "stage": "generating", "message": "در حال تولید پاسخ..."})
        msgs = context_manager.build_messages(
            user_message=user_message,
            language=language,
            short_ctx=short_ctx,
            memories=memories,
            exemplars=exemplars,
        )

        # In Arena Dual Mode -> run Local `qwen3:0.6b` and Online Teacher concurrently for side-by-side comparison!
        if settings.operating_mode == "arena_dual" and settings.teacher.enabled:
            mem_snips = [m["content"] for m in memories]
            local_task = asyncio.create_task(model_manager.chat(msgs, language=language, timeout=35.0))
            teacher_task = asyncio.create_task(
                online_teacher.ask_teacher(user_message, language=language, context_snippets=mem_snips)
            )
            local_res, teacher_res = await asyncio.gather(local_task, teacher_task)
            val = ResponseValidator.validate_chat_response(local_res.content, user_message, language=language)
            local_text = val.cleaned_text if local_res.content else (local_res.error or "بدون پاسخ")
            chosen = teacher_res.content if teacher_res.ok else (
                val.cleaned_text if val.valid else ResponseValidator.build_safe_fallback(language)
            )
            if teacher_res.ok and settings.teacher.auto_distill_exemplars and not memories:
                long_memory.save_teacher_exemplar(
                    user_prompt=user_message,
                    ideal_response=teacher_res.content,
                    language=language,
                    teacher_source=teacher_res.provider,
                )
            arena_meta = {
                "local_model": settings.active_model,
                "local_output": local_text,
                "local_valid": val.valid,
                "local_latency_ms": local_res.latency_ms,
                "teacher_provider": teacher_res.provider,
                "teacher_output": teacher_res.content if teacher_res.ok else teacher_res.error,
                "teacher_ok": teacher_res.ok,
                "teacher_latency_ms": teacher_res.latency_ms,
            }
            return chosen, ("memory" if memories else "arena_dual"), arena_meta

        # Zero-Freeze Smart Hybrid Path:
        # On constrained PCs in `hybrid_teacher` mode, if the query is in Persian and local SQLite has
        # no distilled exemplar for this topic yet, ask the Online Teacher first (0% local CPU spike!)
        # and distill the verified answer into SQLite so `qwen3:0.6b` learns it for future/offline turns!
        if (
            settings.operating_mode == "hybrid_teacher"
            and settings.teacher.enabled
            and language == "fa"
            and not exemplars
        ):
            t_fast = await online_teacher.ask_teacher(
                user_message,
                language=language,
                context_snippets=[m["content"] for m in memories] if memories else None,
            )
            if t_fast.ok:
                if settings.teacher.auto_distill_exemplars and not memories:
                    long_memory.save_teacher_exemplar(
                        user_prompt=user_message,
                        ideal_response=t_fast.content,
                        language=language,
                        teacher_source=t_fast.provider,
                    )
                return t_fast.content, ("memory" if memories else "hybrid_teacher"), None

        # Standard Local-First Execution (`local_strict`, English, or Exemplar-Guided `hybrid_teacher`)
        local_res = await model_manager.chat(msgs, language=language, timeout=35.0)
        val = ResponseValidator.validate_chat_response(
            local_res.content, user_message, language=language, executed_tools=executed_tools
        )

        # If grounded in user memory and local model struggled with Persian syntax, verify memory accuracy!
        if memories and (not local_res.ok or not val.valid):
            if settings.operating_mode == "hybrid_teacher" and settings.teacher.enabled:
                t_res = await online_teacher.ask_teacher(
                    user_message,
                    language=language,
                    context_snippets=[m["content"] for m in memories],
                )
                if t_res.ok:
                    return t_res.content, "memory", None
            mem_bullets = "\n".join(f"- {m['content']}" for m in memories)
            prefix = "بر اساس اطلاعات ذخیره‌شده در حافظه شما:" if language == "fa" else "From your stored memory:"
            return f"{prefix}\n{mem_bullets}", "memory", None

        if local_res.ok and val.valid:
            return val.cleaned_text, ("memory" if memories else "local_model"), None

        # Local `qwen3:0.6b` output failed quality/fluency check ->
        # In `hybrid_teacher` mode, ask Free Online Teacher AND distill into local exemplar memory!
        if settings.operating_mode == "hybrid_teacher" and settings.teacher.enabled:
            await self._emit(
                event_callback,
                {"type": "status", "stage": "teacher_assist", "message": "تقویت پاسخ فارسی و آموزش حافظه محلی..."},
            )
            t_res = await online_teacher.ask_teacher(
                user_message,
                language=language,
                context_snippets=[m["content"] for m in memories] if memories else None,
            )
            if t_res.ok:
                if settings.teacher.auto_distill_exemplars and not memories:
                    long_memory.save_teacher_exemplar(
                        user_prompt=user_message,
                        ideal_response=t_res.content,
                        language=language,
                        teacher_source=t_res.provider,
                    )
                return t_res.content, ("memory" if memories else "hybrid_teacher"), None

        # Strict offline fallback: honest, safe, non-deceptive!
        reason = "hallucinated_action" if val.hallucinated_action else "low_confidence"
        if not local_res.ok and local_res.error:
            audit_logger.log(
                "llm_inference_failed",
                "model_manager",
                "error",
                details={"error": local_res.error, "issues": val.issues},
            )
            if language == "fa":
                return f"خطا در ارتباط با مدل محلی Ollama: {local_res.error}", "local_model", None
            return f"Local Ollama inference error: {local_res.error}", "local_model", None

        audit_logger.log(
            "llm_output_rejected",
            "response_validator",
            "fallback",
            details={"issues": val.issues, "score": val.quality_score},
        )
        return ResponseValidator.build_safe_fallback(language, reason=reason), "local_model", None

    @staticmethod
    def _describe_confirmation(decision: RouteDecision) -> tuple[str, str]:
        if decision.action == "memory_save":
            c = decision.arguments.get("content", "")
            return (
                f"آیا این مورد را در حافظه بلندمدت ذخیره کنم؟ «{c}»",
                f"Should I save this to your persistent memory? \"{c}\"",
            )
        if decision.action == "file_delete":
            p = decision.arguments.get("path", "")
            return (
                f"آیا از حذف فایل «{p}» مطمئن هستید؟ (قبل از حذف، نسخه پشتیبان ایجاد می‌شود.)",
                f"Are you sure you want to delete '{p}'? (A backup copy will be created first.)",
            )
        return (
            f"آیا اجرای ابزار حساس «{decision.action}» را تأیید می‌کنید؟",
            f"Do you approve running '{decision.action}'?",
        )


luna_agent = LunaAgent()
