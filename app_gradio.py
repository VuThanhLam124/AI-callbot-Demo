#!/usr/bin/env python3
"""Gradio interface for VNPost telecom callbot demo."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import requests

from app.callbot import TelecomCallbot
from app.tool_system import TelecomToolSystem

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency 'gradio'. Install requirements-gradio.txt."
    ) from exc


@dataclass
class ASRSettings:
    model_path: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    no_speech_threshold: float
    log_prob_threshold: float


ASR_CACHE: dict[tuple[str, str, str], Any] = {}
TOOL_SYSTEM = TelecomToolSystem()


def _normalize_vllm_base_url(base_url: str) -> str:
    url = base_url.strip()
    if not url:
        return "http://localhost:8002/v1"
    if url.endswith("/"):
        url = url[:-1]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _call_vllm(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    endpoint = f"{_normalize_vllm_base_url(base_url)}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # Keep latency low for callbot behavior.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(endpoint, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _fallback_reply(user_text: str, phone: str) -> str:
    bot = TelecomCallbot(mode="inbound", phone=phone or "0987000001")
    return bot.reply(user_text)


def _ensure_system_message(
    llm_messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    messages = list(llm_messages or [])
    if not messages:
        return [{"role": "system", "content": system_prompt}]

    if messages[0].get("role") != "system":
        return [{"role": "system", "content": system_prompt}, *messages]

    messages[0]["content"] = system_prompt
    return messages


def _chat_once(
    user_text: str,
    chat_history: list[dict[str, str]] | None,
    llm_messages: list[dict[str, str]] | None,
    tool_state: dict | None,
    base_url: str,
    model: str,
    system_prompt: str,
    phone: str,
    use_vllm: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, list[dict[str, str]], list[dict[str, str]], dict, str]:
    text = (user_text or "").strip()
    history = list(chat_history or [])
    messages = _ensure_system_message(list(llm_messages or []), system_prompt)
    state = dict(tool_state or {})

    if not text:
        return "", history, messages, state, "Vui lòng nhập hoặc chuyển giọng nói thành văn bản trước."

    history.append({"role": "user", "content": text})
    messages.append({"role": "user", "content": text})

    handled, tool_reply, state, tool_name = TOOL_SYSTEM.handle(text, phone, state)
    if handled:
        assistant = tool_reply
        status = f"Đã xử lý bằng tool: {tool_name}"
    else:
        status = "Đang dùng fallback rule-based."
        if use_vllm:
            try:
                assistant = _call_vllm(
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                if not assistant:
                    assistant = "Xin lỗi, tôi chưa nghe rõ. Bạn có thể nói lại giúp tôi không?"
                status = "vLLM phản hồi thành công (enable_thinking=false)."
            except Exception as exc:
                assistant = _fallback_reply(text, phone)
                status = f"vLLM không khả dụng, chuyển fallback ({type(exc).__name__})."
        else:
            assistant = _fallback_reply(text, phone)

    history.append({"role": "assistant", "content": assistant})
    messages.append({"role": "assistant", "content": assistant})
    return "", history, messages, state, status


def _get_asr_model(settings: ASRSettings) -> Any:
    key = (settings.model_path, settings.device, settings.compute_type)
    if key in ASR_CACHE:
        return ASR_CACHE[key]

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Install requirements-gradio.txt."
        ) from exc

    model = WhisperModel(
        settings.model_path,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    ASR_CACHE[key] = model
    return model


def _transcribe_audio(
    audio_path: str | None,
    asr_model_path: str,
    asr_device: str,
    asr_compute_type: str,
    asr_language: str,
    asr_beam_size: int,
    asr_no_speech_threshold: float,
    asr_log_prob_threshold: float,
) -> tuple[str, str]:
    if not audio_path:
        return "", "Không có audio đầu vào."

    settings = ASRSettings(
        model_path=asr_model_path,
        device=asr_device,
        compute_type=asr_compute_type,
        language=asr_language,
        beam_size=asr_beam_size,
        no_speech_threshold=asr_no_speech_threshold,
        log_prob_threshold=asr_log_prob_threshold,
    )

    try:
        model = _get_asr_model(settings)
        segments, _ = model.transcribe(
            audio_path,
            language=settings.language,
            beam_size=settings.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=settings.no_speech_threshold,
            log_prob_threshold=settings.log_prob_threshold,
            temperature=0.0,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return "", "ASR hoàn tất nhưng không nhận được nội dung."
        return text, "ASR hoàn tất."
    except Exception as exc:
        return "", f"ASR lỗi: {type(exc).__name__}"


def _reset_conversation() -> tuple[list[dict[str, str]], list[dict[str, str]], dict, str, str]:
    return [], [], {}, "", "Đã xóa lịch sử hội thoại."


def build_ui() -> gr.Blocks:
    default_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1")
    default_model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-1.7B-GPTQ-Int8")
    default_asr_path = os.getenv("ASR_MODEL_PATH", "models/PhoWhisper-small-ct2")

    with gr.Blocks(title="VNPost Telecom Callbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # VNPost Telecom Callbot (Gradio)
            Demo văn bản + giọng nói với:
            - ASR: PhoWhisper-small (CTranslate2 via faster-whisper)
            - LLM: vLLM OpenAI-compatible API
            - Yêu cầu chat luôn dùng `enable_thinking=false`
            - Có tool system bám theo sample data để trả lời logic hơn
            """
        )

        status = gr.Textbox(label="Trạng thái", value="Sẵn sàng.", interactive=False)
        chat_history = gr.Chatbot(label="Hội thoại", type="messages", height=420)
        llm_state = gr.State([])
        tool_state = gr.State({})

        with gr.Accordion("Cấu hình runtime", open=False):
            with gr.Row():
                base_url = gr.Textbox(label="vLLM Base URL", value=default_base_url)
                model_name = gr.Textbox(label="vLLM Model", value=default_model)
            with gr.Row():
                use_vllm = gr.Checkbox(label="Dùng vLLM", value=True)
                phone = gr.Textbox(label="Số thuê bao demo", value="0987000001")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top P")
                max_tokens = gr.Slider(32, 512, value=160, step=16, label="Max Tokens")
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=(
                    "Bạn là trợ lý giọng nói VNPost Telecom. "
                    "Luôn trả lời tiếng Việt có dấu, ngắn gọn, rõ ràng. "
                    "Ưu tiên tư vấn gói cước, ưu đãi SIM và tra cứu thuê bao."
                ),
                lines=3,
            )
            with gr.Row():
                asr_model_path = gr.Textbox(label="ASR Model Path", value=default_asr_path)
                asr_device = gr.Dropdown(label="ASR Device", choices=["cuda", "cpu"], value="cuda")
                asr_compute_type = gr.Dropdown(
                    label="ASR Compute Type",
                    choices=["int8_float16", "float16", "int8", "float32"],
                    value="int8_float16",
                )
            with gr.Row():
                asr_language = gr.Textbox(label="ASR Language", value="vi")
                asr_beam_size = gr.Slider(1, 5, value=2, step=1, label="ASR Beam Size")
                asr_no_speech_threshold = gr.Slider(
                    0.0,
                    1.0,
                    value=0.6,
                    step=0.05,
                    label="ASR No Speech Threshold",
                )
                asr_log_prob_threshold = gr.Slider(
                    -3.0,
                    0.0,
                    value=-1.0,
                    step=0.1,
                    label="ASR Log Prob Threshold",
                )

        with gr.Tabs():
            with gr.Tab("Nhập văn bản"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Tin nhắn người dùng",
                        lines=2,
                        placeholder="Nhập câu hỏi về gói cước, mua SIM, ưu đãi...",
                    )
                with gr.Row():
                    send_btn = gr.Button("Gửi", variant="primary")
                    clear_btn = gr.Button("Xóa hội thoại")

            with gr.Tab("Nhập giọng nói"):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Ghi âm hoặc tải file audio",
                )
                with gr.Row():
                    transcribe_btn = gr.Button("Chuyển giọng nói thành văn bản")
                    send_transcript_btn = gr.Button("Gửi transcript", variant="primary")
                transcript_box = gr.Textbox(label="Transcript", lines=3)

        common_inputs = [
            chat_history,
            llm_state,
            tool_state,
            base_url,
            model_name,
            system_prompt,
            phone,
            use_vllm,
            temperature,
            top_p,
            max_tokens,
        ]

        send_btn.click(
            fn=lambda user_text, *args: _chat_once(user_text, *args),
            inputs=[text_input, *common_inputs],
            outputs=[text_input, chat_history, llm_state, tool_state, status],
        )
        text_input.submit(
            fn=lambda user_text, *args: _chat_once(user_text, *args),
            inputs=[text_input, *common_inputs],
            outputs=[text_input, chat_history, llm_state, tool_state, status],
        )

        transcribe_btn.click(
            fn=_transcribe_audio,
            inputs=[
                audio_input,
                asr_model_path,
                asr_device,
                asr_compute_type,
                asr_language,
                asr_beam_size,
                asr_no_speech_threshold,
                asr_log_prob_threshold,
            ],
            outputs=[transcript_box, status],
        )
        send_transcript_btn.click(
            fn=lambda user_text, *args: _chat_once(user_text, *args),
            inputs=[transcript_box, *common_inputs],
            outputs=[transcript_box, chat_history, llm_state, tool_state, status],
        )

        clear_btn.click(
            fn=_reset_conversation,
            inputs=[],
            outputs=[chat_history, llm_state, tool_state, text_input, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gradio interface for telecom callbot.")
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        help="Server port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        help="Enable public share URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
