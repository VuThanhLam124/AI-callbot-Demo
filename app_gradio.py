#!/usr/bin/env python3
"""Gradio interface for VNPost telecom callbot demo."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

import requests

from app.callbot import TelecomCallbot
from app.llm_tool_agent import LLMToolAgent
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


@dataclass
class RealtimeConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    min_speech_ms: int = 240
    endpoint_silence_ms: int = 520
    speech_rms_threshold: float = 0.012

    @property
    def bytes_per_frame(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0) * 2)

    @property
    def min_speech_frames(self) -> int:
        return max(1, self.min_speech_ms // self.frame_ms)

    @property
    def endpoint_silence_frames(self) -> int:
        return max(1, self.endpoint_silence_ms // self.frame_ms)


def _pcm16_rms(pcm16_bytes: bytes) -> float:
    if not pcm16_bytes:
        return 0.0
    samples = memoryview(pcm16_bytes).cast("h")
    if not samples:
        return 0.0
    mean_square = sum(int(s) * int(s) for s in samples) / len(samples)
    return math.sqrt(mean_square) / 32768.0


class RealtimeTurnDetector:
    """Lightweight endpointing for Gradio audio stream chunks."""

    def __init__(self, cfg: RealtimeConfig):
        self.cfg = cfg
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buffer = bytearray()
        self._tail = bytearray()

    def reset(self) -> None:
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buffer.clear()
        self._tail.clear()

    def feed(self, pcm16_chunk: bytes) -> list[bytes]:
        if not pcm16_chunk:
            return []
        data = bytes(self._tail) + pcm16_chunk
        frame_size = self.cfg.bytes_per_frame
        finalized: list[bytes] = []

        full_len = (len(data) // frame_size) * frame_size
        self._tail = bytearray(data[full_len:])

        for start in range(0, full_len, frame_size):
            frame = data[start : start + frame_size]
            rms = _pcm16_rms(frame)
            is_speech = rms >= self.cfg.speech_rms_threshold

            if is_speech:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_frames = 0
                    self._silence_frames = 0
                    self._buffer.clear()
                self._speech_frames += 1
                self._silence_frames = 0
                self._buffer.extend(frame)
                continue

            if not self._in_speech:
                continue

            self._silence_frames += 1
            self._buffer.extend(frame)
            if self._silence_frames < self.cfg.endpoint_silence_frames:
                continue

            if self._speech_frames >= self.cfg.min_speech_frames:
                finalized.append(bytes(self._buffer))
            self._in_speech = False
            self._speech_frames = 0
            self._silence_frames = 0
            self._buffer.clear()

        return finalized


ASR_CACHE: dict[tuple[str, str, str], Any] = {}
ASR_PREPARED_PATHS: set[str] = set()
ASR_PREP_LOCK = threading.Lock()
TOOL_SYSTEM = TelecomToolSystem()
TOOL_AGENT = LLMToolAgent(TOOL_SYSTEM)
CHATBOT_SUPPORTS_TYPE = "type" in inspect.signature(gr.Chatbot.__init__).parameters
CHATBOT_MODE = os.getenv("GRADIO_CHATBOT_MODE", "messages").strip().lower()
if CHATBOT_MODE not in {"messages", "tuples"}:
    CHATBOT_MODE = "messages"
AUDIO_SUPPORTS_AUTOPLAY = "autoplay" in inspect.signature(gr.Audio.__init__).parameters
TTS_CACHE: dict[tuple[str, str, str, str, str], tuple[Any, Any]] = {}
TTS_LOCK = threading.Lock()
VIENEU_AVAILABLE = importlib.util.find_spec("vieneu") is not None


def _new_realtime_state() -> dict[str, Any]:
    return {
        "detector": None,
        "cfg_key": None,
    }


def _try_prepare_ct2_model(model_path: str) -> tuple[bool, str]:
    """Prepare local CTranslate2 model directory when missing."""
    normalized = os.path.normpath(model_path)
    if os.path.isdir(normalized):
        return True, ""
    if not (
        normalized.startswith("models" + os.path.sep)
        or normalized.startswith("." + os.path.sep + "models" + os.path.sep)
    ):
        return False, ""

    with ASR_PREP_LOCK:
        if os.path.isdir(normalized):
            return True, ""
        if normalized in ASR_PREPARED_PATHS:
            return False, f"ASR path '{normalized}' vẫn chưa sẵn sàng sau lần chuẩn bị trước."
        ASR_PREPARED_PATHS.add(normalized)

        source_model = os.getenv("PHOWHISPER_SOURCE_MODEL", "vinai/PhoWhisper-small").strip()
        quantization = os.getenv("PHOWHISPER_CT2_QUANTIZATION", "int8_float16").strip()
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "convert_phowhisper_ct2.sh")
        os.makedirs(os.path.dirname(normalized) or ".", exist_ok=True)

        if os.path.isfile(script_path):
            cmd = ["bash", script_path, source_model, normalized, quantization]
        else:
            cmd = [
                "ct2-transformers-converter",
                "--model",
                source_model,
                "--output_dir",
                normalized,
                "--copy_files",
                "tokenizer.json",
                "preprocessor_config.json",
                "--quantization",
                quantization,
            ]
        print(f"[INFO] Auto-prepare ASR model: {' '.join(cmd)}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            return False, f"Auto-convert ASR thất bại (code={proc.returncode}). {err_tail}"
        if os.path.isdir(normalized):
            return True, ""
        return False, f"Auto-convert hoàn tất nhưng không thấy thư mục '{normalized}'."


def _resolve_asr_model_path(model_path: str) -> tuple[str, str]:
    requested = (model_path or "").strip()
    if not requested:
        return "small", "ASR path trống, fallback sang model 'small'."
    if os.path.isdir(requested):
        return requested, ""
    if os.path.sep in requested:
        try:
            ok, prep_msg = _try_prepare_ct2_model(requested)
        except Exception as exc:
            ok = False
            prep_msg = f"Auto-convert ASR lỗi: {type(exc).__name__}"
        if ok and os.path.isdir(requested):
            return requested, ""
        fallback_model = os.getenv("ASR_FALLBACK_MODEL", "small").strip() or "small"
        if prep_msg:
            msg = (
                f"{prep_msg} Fallback sang '{fallback_model}'. "
                "Bạn có thể tự convert bằng ./scripts/convert_phowhisper_ct2.sh."
            )
            return fallback_model, msg
        return fallback_model, f"ASR path '{requested}' không tồn tại, fallback sang '{fallback_model}'."
    return requested, ""


def _get_vieneu_engine_and_voice(
    backbone_repo: str,
    backbone_device: str,
    codec_repo: str,
    codec_device: str,
    voice_id: str,
) -> tuple[Any, Any]:
    key = (
        backbone_repo.strip(),
        backbone_device.strip(),
        codec_repo.strip(),
        codec_device.strip(),
        voice_id.strip(),
    )
    with TTS_LOCK:
        if key in TTS_CACHE:
            return TTS_CACHE[key]
        try:
            from vieneu import Vieneu  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Thiếu package 'vieneu'. Cài thêm: "
                "pip install vieneu --extra-index-url "
                "https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/"
            ) from exc

        engine = Vieneu(
            backbone_repo=key[0],
            backbone_device=key[1] or "cpu",
            codec_repo=key[2],
            codec_device=key[3] or "cpu",
        )
        voice = None
        if key[4]:
            try:
                voice = engine.get_preset_voice(key[4])
            except Exception:
                voice = None
        if voice is None:
            try:
                voice = engine.get_preset_voice(None)
            except Exception:
                voice = None
        TTS_CACHE[key] = (engine, voice)
        return engine, voice


def _synthesize_vieneu_audio(
    text: str,
    backbone_repo: str,
    backbone_device: str,
    codec_repo: str,
    codec_device: str,
    voice_id: str,
) -> tuple[int, Any]:
    if not text.strip():
        raise RuntimeError("Không có nội dung để TTS.")
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Thiếu package 'numpy'.") from exc

    engine, voice = _get_vieneu_engine_and_voice(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
        voice_id=voice_id,
    )
    kwargs: dict[str, Any] = {"text": text}
    if voice is not None:
        kwargs["voice"] = voice
    audio = engine.infer(**kwargs)
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("VieNeu trả về audio rỗng.")
    arr = np.clip(arr, -1.0, 1.0)
    sample_rate = int(getattr(engine, "sample_rate", 24000))
    return sample_rate, arr


def _append_chat_history(
    history: list[Any],
    user_text: str,
    assistant_text: str,
) -> list[Any]:
    if CHATBOT_MODE == "messages":
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
    else:
        # Legacy Gradio tuple format: (user, assistant)
        history.append((user_text, assistant_text))
    return history


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
    if resp.status_code >= 400:
        raise RuntimeError(f"vLLM HTTP {resp.status_code}: {resp.text[:400]}")
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
    chat_history: list[Any] | None,
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
) -> tuple[str, list[Any], list[dict[str, str]], dict, str]:
    text = (user_text or "").strip()
    history = list(chat_history or [])
    messages = _ensure_system_message(list(llm_messages or []), system_prompt)
    state = dict(tool_state or {})

    if not text:
        return "", history, messages, state, "Vui lòng nhập hoặc chuyển giọng nói thành văn bản trước."

    messages.append({"role": "user", "content": text})

    if use_vllm:
        try:
            assistant, messages, state, used_tools = TOOL_AGENT.run_turn(
                messages=messages,
                phone=phone,
                state=state,
                llm_chat=lambda chat_messages: _call_vllm(
                    base_url=base_url,
                    model=model,
                    messages=chat_messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ),
            )
            if not assistant:
                assistant = "Xin lỗi, tôi chưa nghe rõ. Bạn có thể nói lại giúp tôi không?"
                messages.append({"role": "assistant", "content": assistant})

            if used_tools:
                status = "vLLM phản hồi thành công. Tools: " + ", ".join(used_tools)
            else:
                status = "vLLM phản hồi thành công. Không cần gọi tool."
        except Exception as exc:
            assistant = _fallback_reply(text, phone)
            messages.append({"role": "assistant", "content": assistant})
            status = f"vLLM không khả dụng, chuyển fallback ({type(exc).__name__})."
    else:
        assistant = _fallback_reply(text, phone)
        messages.append({"role": "assistant", "content": assistant})
        status = "Đang dùng fallback rule-based."

    history = _append_chat_history(history, text, assistant)
    return "", history, messages, state, status


def _get_asr_model(settings: ASRSettings) -> Any:
    resolved_model_path, resolve_msg = _resolve_asr_model_path(settings.model_path)
    if resolve_msg:
        print(f"[WARN] {resolve_msg}")

    key = (resolved_model_path, settings.device, settings.compute_type)
    if key in ASR_CACHE:
        return ASR_CACHE[key]

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Install requirements-gradio.txt."
        ) from exc

    try:
        model = WhisperModel(
            resolved_model_path,
            device=settings.device,
            compute_type=settings.compute_type,
        )
    except Exception as exc:
        guidance = (
            "ASR model không tải được. Nếu dùng PhoWhisper local, hãy convert trước bằng "
            "`./scripts/convert_phowhisper_ct2.sh` rồi đặt đúng `ASR Model Path`. "
            "Hoặc đặt tạm `ASR Model Path = small` để demo nhanh."
        )
        raise RuntimeError(f"{guidance} (model='{resolved_model_path}')") from exc

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
        err = str(exc).strip().replace("\n", " ")
        return "", f"ASR lỗi: {type(exc).__name__}. {err[:220]}"


def _resample_audio(audio: Any, source_sr: int, target_sr: int) -> Any:
    if source_sr == target_sr:
        return audio
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'.") from exc
    if len(audio) == 0:
        return audio
    target_len = max(1, int(len(audio) * target_sr / source_sr))
    src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _chunk_to_pcm16(audio_chunk: Any, target_sr: int = 16000) -> bytes:
    if audio_chunk is None:
        return b""
    if not isinstance(audio_chunk, (tuple, list)) or len(audio_chunk) != 2:
        return b""

    source_sr, data = audio_chunk
    if data is None:
        return b""

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'.") from exc

    arr = np.asarray(data)
    if arr.size == 0:
        return b""

    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32).reshape(-1)

    # Browser audio from Gradio is usually float in [-1, 1]. Convert robustly.
    if np.max(np.abs(arr)) > 1.5:
        arr = arr / 32768.0

    sr = int(source_sr)
    if sr <= 0:
        sr = target_sr
    if sr != target_sr:
        arr = _resample_audio(arr, sr, target_sr)

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _transcribe_pcm16(
    pcm16_bytes: bytes,
    settings: ASRSettings,
) -> str:
    if not pcm16_bytes:
        return ""
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'.") from exc

    model = _get_asr_model(settings)
    audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio,
        language=settings.language,
        beam_size=settings.beam_size,
        vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=settings.no_speech_threshold,
        log_prob_threshold=settings.log_prob_threshold,
        temperature=0.0,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _realtime_stream_step(
    audio_chunk: Any,
    realtime_state: dict[str, Any] | None,
    chat_history: list[Any] | None,
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
    asr_model_path: str,
    asr_device: str,
    asr_compute_type: str,
    asr_language: str,
    asr_beam_size: int,
    asr_no_speech_threshold: float,
    asr_log_prob_threshold: float,
    realtime_enabled: bool,
    realtime_rms_threshold: float,
    realtime_min_speech_ms: int,
    realtime_endpoint_silence_ms: int,
    tts_enabled: bool,
    tts_backbone_repo: str,
    tts_backbone_device: str,
    tts_codec_repo: str,
    tts_codec_device: str,
    tts_voice_id: str,
) -> tuple[dict[str, Any], list[Any], list[dict[str, str]], dict, str, str, Any]:
    history = list(chat_history or [])
    messages = list(llm_messages or [])
    state = dict(tool_state or {})
    rt = dict(realtime_state or _new_realtime_state())
    bot_audio: Any = None

    if not realtime_enabled:
        rt["detector"] = None
        return rt, history, messages, state, "", "Realtime mic đang tắt.", bot_audio

    cfg = RealtimeConfig(
        sample_rate=16000,
        frame_ms=20,
        min_speech_ms=int(realtime_min_speech_ms),
        endpoint_silence_ms=int(realtime_endpoint_silence_ms),
        speech_rms_threshold=float(realtime_rms_threshold),
    )
    cfg_key = (
        cfg.sample_rate,
        cfg.frame_ms,
        cfg.min_speech_ms,
        cfg.endpoint_silence_ms,
        round(cfg.speech_rms_threshold, 4),
    )
    detector = rt.get("detector")
    if detector is None or rt.get("cfg_key") != cfg_key:
        detector = RealtimeTurnDetector(cfg)
        rt["detector"] = detector
        rt["cfg_key"] = cfg_key

    pcm_chunk = _chunk_to_pcm16(audio_chunk, target_sr=cfg.sample_rate)
    if not pcm_chunk:
        return rt, history, messages, state, "", "Đang nghe realtime...", bot_audio

    asr_settings = ASRSettings(
        model_path=asr_model_path,
        device=asr_device,
        compute_type=asr_compute_type,
        language=asr_language,
        beam_size=int(asr_beam_size),
        no_speech_threshold=float(asr_no_speech_threshold),
        log_prob_threshold=float(asr_log_prob_threshold),
    )

    final_utterances = detector.feed(pcm_chunk)
    if not final_utterances:
        return rt, history, messages, state, "", "Đang nghe realtime...", bot_audio

    latest_transcript = ""
    latest_status = "Đang nghe realtime..."
    for utterance in final_utterances:
        try:
            user_text = _transcribe_pcm16(utterance, asr_settings).strip()
        except Exception as exc:
            err = str(exc).strip().replace("\n", " ")
            return (
                rt,
                history,
                messages,
                state,
                "",
                f"ASR realtime lỗi: {type(exc).__name__}. {err[:220]}",
                bot_audio,
            )
        if not user_text:
            continue
        latest_transcript = user_text
        _, history, messages, state, latest_status = _chat_once(
            user_text=user_text,
            chat_history=history,
            llm_messages=messages,
            tool_state=state,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            phone=phone,
            use_vllm=use_vllm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
        )
        if tts_enabled:
            assistant_text = ""
            if messages and messages[-1].get("role") == "assistant":
                assistant_text = messages[-1].get("content", "")
            if assistant_text:
                try:
                    bot_audio = _synthesize_vieneu_audio(
                        text=assistant_text,
                        backbone_repo=tts_backbone_repo,
                        backbone_device=tts_backbone_device,
                        codec_repo=tts_codec_repo,
                        codec_device=tts_codec_device,
                        voice_id=tts_voice_id,
                    )
                except Exception as exc:
                    latest_status = f"{latest_status} | TTS lỗi: {type(exc).__name__}"
                    bot_audio = None

    if latest_transcript:
        return rt, history, messages, state, latest_transcript, latest_status, bot_audio
    return rt, history, messages, state, "", "ASR realtime chưa nhận được câu rõ ràng.", bot_audio


def _reset_conversation() -> tuple[
    list[Any],
    list[dict[str, str]],
    dict,
    str,
    dict[str, Any],
    str,
    Any,
]:
    return [], [], {}, "Đã xóa lịch sử hội thoại.", _new_realtime_state(), "", None


def build_ui() -> gr.Blocks:
    default_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1")
    default_model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-1.7B-GPTQ-Int8")
    default_asr_path = os.getenv("ASR_MODEL_PATH", "models/PhoWhisper-small-ct2")

    with gr.Blocks(title="VNPost Telecom Callbot") as demo:
        gr.Markdown(
            """
            # VNPost Telecom Callbot (Gradio)
            Demo realtime microphone với:
            - ASR: PhoWhisper-small (CTranslate2 via faster-whisper)
            - LLM: vLLM OpenAI-compatible API
            - Yêu cầu chat luôn dùng `enable_thinking=false`
            - LLM tự quyết định gọi tool để lấy dữ liệu mẫu trước khi trả lời
            """
        )

        status = gr.Textbox(label="Trạng thái", value="Sẵn sàng.", interactive=False)
        chatbot_kwargs: dict[str, Any] = {"label": "Hội thoại", "height": 420, "allow_tags": False}
        if CHATBOT_SUPPORTS_TYPE:
            chatbot_kwargs["type"] = "messages"
        chat_history = gr.Chatbot(**chatbot_kwargs)
        llm_state = gr.State([])
        tool_state = gr.State({})
        realtime_state = gr.State(_new_realtime_state())

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
                max_tokens = gr.Slider(32, 512, value=128, step=16, label="Max Tokens")
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=(
                    "Bạn là trợ lý giọng nói VNPost Telecom. "
                    "Luôn trả lời tiếng Việt có dấu, ngắn gọn, rõ ràng. "
                    "Ưu tiên tư vấn gói cước, ưu đãi SIM và tra cứu thuê bao. "
                    "Khi cần dữ liệu giá/số lượng/thông tin thuê bao, hãy gọi tool trước rồi mới trả lời."
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

        gr.Markdown("## Realtime mic (tự gửi khi bạn dừng nói)")
        realtime_enabled = gr.Checkbox(
            label="Bật realtime voice",
            value=True,
            info="Bot tự nhận tiếng nói và tự phản hồi, không có turn-based input.",
        )
        with gr.Row():
            realtime_rms_threshold = gr.Slider(
                0.005,
                0.06,
                value=0.012,
                step=0.001,
                label="Realtime RMS threshold",
            )
            realtime_min_speech_ms = gr.Slider(
                120,
                800,
                value=240,
                step=20,
                label="Realtime Min Speech (ms)",
            )
            realtime_endpoint_silence_ms = gr.Slider(
                250,
                1400,
                value=520,
                step=20,
                label="Realtime Endpoint Silence (ms)",
            )
        realtime_audio = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Realtime microphone stream",
        )
        realtime_transcript = gr.Textbox(
            label="Transcript realtime (auto)",
            lines=2,
            interactive=False,
        )
        gr.Markdown("### Realtime bot voice")
        realtime_tts_enabled = gr.Checkbox(
            label="Bật bot trả lời bằng giọng nói",
            value=VIENEU_AVAILABLE,
            info=(
                "Dùng VieNeu-TTS để phát audio bot ngay trong trình duyệt."
                if VIENEU_AVAILABLE
                else "Chưa có package vieneu. Cài thêm rồi bật checkbox này."
            ),
        )
        with gr.Row():
            realtime_tts_backbone_repo = gr.Textbox(
                label="TTS Backbone Repo",
                value="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
            )
            realtime_tts_backbone_device = gr.Dropdown(
                label="TTS Backbone Device",
                choices=["cpu", "cuda", "mps"],
                value="cpu",
            )
        with gr.Row():
            realtime_tts_codec_repo = gr.Textbox(
                label="TTS Codec Repo",
                value="neuphonic/distill-neucodec",
            )
            realtime_tts_codec_device = gr.Dropdown(
                label="TTS Codec Device",
                choices=["cpu", "cuda", "mps"],
                value="cpu",
            )
            realtime_tts_voice_id = gr.Textbox(
                label="TTS Voice ID (optional)",
                value="",
            )
        bot_audio_kwargs: dict[str, Any] = {
            "label": "Audio bot realtime",
            "type": "numpy",
            "interactive": False,
        }
        if AUDIO_SUPPORTS_AUTOPLAY:
            bot_audio_kwargs["autoplay"] = True
        realtime_bot_audio = gr.Audio(**bot_audio_kwargs)
        clear_btn = gr.Button("Xóa hội thoại")

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

        realtime_audio.stream(
            fn=_realtime_stream_step,
            inputs=[
                realtime_audio,
                realtime_state,
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
                asr_model_path,
                asr_device,
                asr_compute_type,
                asr_language,
                asr_beam_size,
                asr_no_speech_threshold,
                asr_log_prob_threshold,
                realtime_enabled,
                realtime_rms_threshold,
                realtime_min_speech_ms,
                realtime_endpoint_silence_ms,
                realtime_tts_enabled,
                realtime_tts_backbone_repo,
                realtime_tts_backbone_device,
                realtime_tts_codec_repo,
                realtime_tts_codec_device,
                realtime_tts_voice_id,
            ],
            outputs=[
                realtime_state,
                chat_history,
                llm_state,
                tool_state,
                realtime_transcript,
                status,
                realtime_bot_audio,
            ],
        )

        clear_btn.click(
            fn=_reset_conversation,
            inputs=[],
            outputs=[
                chat_history,
                llm_state,
                tool_state,
                status,
                realtime_state,
                realtime_transcript,
                realtime_bot_audio,
            ],
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
