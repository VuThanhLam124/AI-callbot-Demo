"""Realtime callbot pipeline with VAD, ASR, LLM and barge-in support."""

from __future__ import annotations

import difflib
import math
import queue
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.llm_tool_agent import LLMToolAgent
from app.tool_system import TelecomToolSystem


def _pcm16_rms(pcm16_bytes: bytes) -> float:
    if not pcm16_bytes:
        return 0.0
    samples = memoryview(pcm16_bytes).cast("h")
    if not samples:
        return 0.0
    mean_square = sum(int(s) * int(s) for s in samples) / len(samples)
    return math.sqrt(mean_square) / 32768.0


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    vad_aggressiveness: int = 2
    min_speech_ms: int = 180
    endpoint_silence_ms: int = 450
    utterance_min_rms: float = 0.010
    barge_in_min_rms: float = 0.015

    @property
    def bytes_per_frame(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0) * 2)

    @property
    def min_speech_frames(self) -> int:
        return max(1, self.min_speech_ms // self.frame_ms)

    @property
    def endpoint_silence_frames(self) -> int:
        return max(1, self.endpoint_silence_ms // self.frame_ms)


@dataclass
class FrameDecision:
    speech_started: bool = False
    speech_active: bool = False
    rms: float = 0.0
    final_audio: bytes | None = None


class StreamingVAD:
    """Incremental VAD for endpointing and barge-in triggers."""

    def __init__(self, cfg: AudioConfig):
        try:
            import webrtcvad  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'webrtcvad-wheels'. Install requirements-realtime.txt."
            ) from exc

        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.buffer = bytearray()

    def reset(self) -> None:
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.buffer.clear()

    def process(self, frame: bytes) -> FrameDecision:
        if len(frame) != self.cfg.bytes_per_frame:
            return FrameDecision()

        rms = _pcm16_rms(frame)
        is_speech = self.vad.is_speech(frame, self.cfg.sample_rate)
        result = FrameDecision(speech_active=is_speech, rms=rms)

        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.speech_frames = 0
                self.silence_frames = 0
                self.buffer.clear()
                result.speech_started = True

            self.speech_frames += 1
            self.silence_frames = 0
            self.buffer.extend(frame)
            return result

        if not self.in_speech:
            return result

        self.silence_frames += 1
        self.buffer.extend(frame)
        if self.silence_frames < self.cfg.endpoint_silence_frames:
            return result

        if self.speech_frames >= self.cfg.min_speech_frames:
            result.final_audio = bytes(self.buffer)

        self.reset()
        return result


class FasterWhisperASR:
    """ASR engine for CTranslate2 Whisper-compatible models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "vi",
        beam_size: int = 2,
        no_speech_threshold: float = 0.6,
        log_prob_threshold: float = -1.0,
    ):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'faster-whisper'. Install requirements-realtime.txt."
            ) from exc

        self.language = language
        self.beam_size = beam_size
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, pcm16_bytes: bytes) -> str:
        if not pcm16_bytes:
            return ""

        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'numpy'. Install requirements-realtime.txt."
            ) from exc

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            temperature=0.0,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text


class VLLMChatClient:
    """OpenAI-compatible client for vLLM server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "dummy",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 160,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'openai'. Install requirements-realtime.txt."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # Required for latency-sensitive callbot behavior.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        content = resp.choices[0].message.content
        return content.strip() if content else ""


class TTSPlayer:
    """Simple TTS playback with interrupt support for barge-in."""

    def __init__(
        self,
        mode: str = "text",
        rate_wpm: int = 180,
        output_device: str | int | None = None,
        vieneu_backbone_repo: str = "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        vieneu_backbone_device: str = "cpu",
        vieneu_codec_repo: str = "neuphonic/distill-neucodec",
        vieneu_codec_device: str = "cpu",
        vieneu_voice_id: str | None = None,
        vieneu_streaming: bool = False,
    ):
        self.mode = mode
        self.rate_wpm = rate_wpm
        self.output_device = output_device
        self.vieneu_backbone_repo = vieneu_backbone_repo
        self.vieneu_backbone_device = vieneu_backbone_device
        self.vieneu_codec_repo = vieneu_codec_repo
        self.vieneu_codec_device = vieneu_codec_device
        self.vieneu_voice_id = vieneu_voice_id
        self.vieneu_streaming = vieneu_streaming
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._engine: Any = None
        self._vieneu_engine: Any = None
        self._vieneu_voice: Any = None
        self._last_started_at = 0.0
        self._last_ended_at = 0.0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def last_started_at(self) -> float:
        return self._last_started_at

    @property
    def last_ended_at(self) -> float:
        return self._last_ended_at

    def speak_async(self, text: str) -> None:
        if not text:
            return
        self.stop()
        self._stop_event.clear()
        self._last_started_at = time.time()
        self._thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self.mode == "pyttsx3" and self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._is_speaking = False

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._vieneu_engine is not None:
                try:
                    close_fn = getattr(self._vieneu_engine, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass
            self._vieneu_engine = None
            self._vieneu_voice = None

    def _speak_worker(self, text: str) -> None:
        self._is_speaking = True
        try:
            try:
                if self.mode == "pyttsx3":
                    self._speak_pyttsx3(text)
                elif self.mode == "vieneu":
                    self._speak_vieneu(text)
                else:
                    self._speak_text_mode(text)
            except Exception as exc:
                print(f"[WARN] TTS backend failed ({type(exc).__name__}). Fallback to text mode.")
                if not self._stop_event.is_set():
                    self._speak_text_mode(text)
        finally:
            self._is_speaking = False
            self._last_ended_at = time.time()
            with self._lock:
                if self.mode == "pyttsx3":
                    self._engine = None

    def _speak_text_mode(self, text: str) -> None:
        print(f"[BOT] {text}")
        words = max(1, len(text.split()))
        total_time = (words / max(80, self.rate_wpm)) * 60.0
        chunk_count = max(1, int(total_time / 0.05))
        for _ in range(chunk_count):
            if self._stop_event.is_set():
                return
            time.sleep(0.05)

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'pyttsx3'. Install requirements-realtime.txt or use --tts-mode text."
            ) from exc

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate_wpm)
        with self._lock:
            self._engine = engine

        engine.say(text)
        engine.runAndWait()

    def _init_vieneu_if_needed(self) -> tuple[Any, Any]:
        with self._lock:
            if self._vieneu_engine is not None:
                return self._vieneu_engine, self._vieneu_voice

            try:
                from vieneu import Vieneu  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Missing dependency 'vieneu'. Install requirements-realtime.txt."
                ) from exc

            self._vieneu_engine = Vieneu(
                backbone_repo=self.vieneu_backbone_repo,
                backbone_device=self.vieneu_backbone_device,
                codec_repo=self.vieneu_codec_repo,
                codec_device=self.vieneu_codec_device,
            )

            voice_id = (self.vieneu_voice_id or "").strip()
            if voice_id:
                try:
                    self._vieneu_voice = self._vieneu_engine.get_preset_voice(voice_id)
                    print(f"[INFO] VieNeu-TTS using voice preset: {voice_id}")
                except Exception:
                    print(
                        f"[WARN] VieNeu-TTS voice '{voice_id}' not found. "
                        "Falling back to model default voice."
                    )
                    self._vieneu_voice = None
            if self._vieneu_voice is None:
                try:
                    self._vieneu_voice = self._vieneu_engine.get_preset_voice(None)
                    print("[INFO] VieNeu-TTS using default preset voice from model.")
                except Exception:
                    self._vieneu_voice = None

            return self._vieneu_engine, self._vieneu_voice

    def _stream_audio_chunk(self, stream: Any, audio_chunk: Any, frame_size: int = 4096) -> None:
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'numpy'. Install requirements-realtime.txt."
            ) from exc

        arr = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        start = 0
        while start < arr.size:
            if self._stop_event.is_set():
                return
            end = min(start + frame_size, arr.size)
            stream.write(arr[start:end].reshape(-1, 1))
            start = end

    def _speak_vieneu(self, text: str) -> None:
        engine, voice = self._init_vieneu_if_needed()
        print(f"[BOT] {text}")

        try:
            import sounddevice as sd  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'sounddevice'. Install requirements-realtime.txt."
            ) from exc

        def _play_with_device(device: str | int | None) -> None:
            sample_rate = int(getattr(engine, "sample_rate", 24000))
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=4096,
                latency="high",
                device=device,
            ) as stream:
                infer_stream = getattr(engine, "infer_stream", None)
                kwargs: dict[str, Any] = {"text": text}
                if voice is not None:
                    kwargs["voice"] = voice

                if self.vieneu_streaming and callable(infer_stream):
                    for chunk in infer_stream(**kwargs):
                        if self._stop_event.is_set():
                            return
                        self._stream_audio_chunk(stream, chunk)
                    return

                audio = engine.infer(**kwargs)
                if self._stop_event.is_set():
                    return
                self._stream_audio_chunk(stream, audio)

        try:
            _play_with_device(self.output_device)
        except Exception:
            if self.output_device is None:
                raise
            print(
                f"[WARN] TTS output-device {self.output_device!r} không tương thích "
                "với sample-rate hiện tại. Fallback sang default output device."
            )
            _play_with_device(None)


class RealtimeCallbot:
    """End-to-end realtime callbot loop with mic input and barge-in."""

    def __init__(
        self,
        asr: FasterWhisperASR,
        llm: VLLMChatClient,
        tts: TTSPlayer,
        audio_cfg: AudioConfig,
        system_prompt: str,
        greeting: str,
        phone: str = "0987000001",
        use_tools: bool = True,
        barge_in_frames: int = 8,
        barge_in_min_bot_ms: int = 250,
        barge_in_enabled: bool = True,
        input_device: str | int | None = None,
        post_tts_guard_ms: int = 1100,
        post_tts_guard_rms: float = 0.030,
    ):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.audio_cfg = audio_cfg
        self.vad = StreamingVAD(audio_cfg)
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.phone = phone
        self.use_tools = use_tools
        self.barge_in_frames = max(1, barge_in_frames)
        self.barge_in_min_bot_ms = max(0, barge_in_min_bot_ms)
        self.barge_in_enabled = barge_in_enabled
        self.input_device = input_device
        self.post_tts_guard_ms = max(0, post_tts_guard_ms)
        self.post_tts_guard_rms = max(0.0, post_tts_guard_rms)
        self.speech_during_tts = 0
        self.stop_event = threading.Event()
        self.audio_q: queue.Queue[bytes] = queue.Queue(maxsize=300)
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self.tool_system = TelecomToolSystem()
        self.tool_agent = LLMToolAgent(self.tool_system)
        self.tool_state: dict[str, Any] = {}
        self.last_bot_text = ""
        self.last_user_text = ""
        self.last_user_at = 0.0
        self.last_barge_in_at = 0.0

    def _on_audio_frame(self, indata: bytes) -> None:
        try:
            self.audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # Drop old frames when overloaded to keep real-time behavior.
            try:
                self.audio_q.get_nowait()
                self.audio_q.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    def run(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'sounddevice'. Install requirements-realtime.txt."
            ) from exc

        frame_samples = int(self.audio_cfg.sample_rate * (self.audio_cfg.frame_ms / 1000.0))

        def callback(indata: Any, frames: int, _time: Any, status: Any) -> None:
            if status:
                print(f"[WARN] Audio status: {status}")
            if frames <= 0:
                return
            self._on_audio_frame(bytes(indata))

        print("[INFO] Starting realtime callbot. Press Ctrl+C to stop.")
        print("[INFO] For cleaner barge-in in local demo, use a headset.")
        self.last_bot_text = self.greeting
        self.tts.speak_async(self.greeting)

        def _run_loop_with_device(device: str | int | None) -> None:
            with sd.RawInputStream(
                samplerate=self.audio_cfg.sample_rate,
                blocksize=frame_samples,
                channels=1,
                dtype="int16",
                device=device,
                callback=callback,
            ):
                while not self.stop_event.is_set():
                    try:
                        frame = self.audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    decision = self.vad.process(frame)
                    self._handle_barge_in(decision)

                    if decision.final_audio is None:
                        continue
                    self._handle_utterance(decision.final_audio)

        try:
            try:
                _run_loop_with_device(self.input_device)
            except Exception:
                if self.input_device is None:
                    raise
                print(
                    f"[WARN] Input-device {self.input_device!r} không tương thích với "
                    f"{self.audio_cfg.sample_rate}Hz/mono. Fallback sang default input device."
                )
                _run_loop_with_device(None)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
        finally:
            self.tts.stop()
            self.tts.close()

    def _handle_barge_in(self, decision: FrameDecision) -> None:
        if not self.barge_in_enabled:
            return

        if not self.tts.is_speaking:
            self.speech_during_tts = 0
            return

        bot_elapsed_ms = (time.time() - self.tts.last_started_at) * 1000.0
        if bot_elapsed_ms < self.barge_in_min_bot_ms:
            self.speech_during_tts = 0
            return

        if decision.speech_active and decision.rms >= self.audio_cfg.barge_in_min_rms:
            self.speech_during_tts += 1
        else:
            self.speech_during_tts = 0

        if self.speech_during_tts >= self.barge_in_frames:
            print("[BARGE-IN] Người dùng chen lời, dừng TTS.")
            self.tts.stop()
            self.last_barge_in_at = time.time()
            # Drop buffered TTS-leakage frames and restart endpointing cleanly.
            self.vad.reset()
            self._drain_audio_queue(max_frames=100)
            self.speech_during_tts = 0

    def _handle_utterance(self, pcm16_audio: bytes) -> None:
        now = time.time()
        utterance_rms = _pcm16_rms(pcm16_audio)
        if utterance_rms < self.audio_cfg.utterance_min_rms:
            return

        elapsed_from_end_ms = (
            (now - self.tts.last_ended_at) * 1000.0 if self.tts.last_ended_at > 0 else 99_999.0
        )
        in_post_tts_guard = elapsed_from_end_ms <= self.post_tts_guard_ms
        if in_post_tts_guard and utterance_rms < self.post_tts_guard_rms:
            # Right after bot speech ends, low-energy chunks are commonly speaker leakage.
            return

        if self.tts.is_speaking and utterance_rms < (self.audio_cfg.barge_in_min_rms * 1.8):
            # While bot is speaking, very-low-energy speech is likely speaker leakage.
            return

        user_text = self.asr.transcribe(pcm16_audio).strip()
        if not user_text or len(user_text) < 2:
            return
        near_barge_in = (now - self.last_barge_in_at) <= 2.0
        near_tts_window = self.tts.is_speaking or elapsed_from_end_ms <= 2400.0 or near_barge_in
        if self._is_likely_echo(user_text, in_post_tts_guard=in_post_tts_guard):
            print(f"[ECHO] Ignored: {user_text}")
            return
        if near_tts_window and self._is_noise_like_text(user_text):
            print(f"[NOISE] Ignored ASR hallucination near TTS: {user_text}")
            return
        if self._is_recent_duplicate(user_text, near_tts_window=near_tts_window, now=now):
            print(f"[DUP] Ignored duplicated near-TTS transcript: {user_text}")
            return

        print(f"[USER] {user_text}")
        self.last_user_text = user_text
        self.last_user_at = now
        self.messages.append({"role": "user", "content": user_text})

        smalltalk_reply = self._smalltalk_reply(user_text)
        if smalltalk_reply:
            self.messages.append({"role": "assistant", "content": smalltalk_reply})
            self.last_bot_text = smalltalk_reply
            self.tts.speak_async(smalltalk_reply)
            return

        assistant_text = ""
        try:
            if self.use_tools:
                assistant_text, self.messages, self.tool_state, used_tools = self.tool_agent.run_turn(
                    messages=self.messages,
                    phone=self.phone,
                    state=self.tool_state,
                    llm_chat=self.llm.chat,
                )
                if used_tools:
                    print(f"[TOOL] LLM đã gọi: {', '.join(used_tools)}")
            else:
                assistant_text = self.llm.chat(self.messages).strip()
        except Exception as exc:
            assistant_text = (
                "Xin lỗi, tôi đang gặp lỗi kết nối AI server. "
                f"Chi tiết: {type(exc).__name__}"
            )

        if not assistant_text:
            assistant_text = "Xin lỗi, tôi chưa nghe rõ. Bạn có thể nói lại giúp tôi không?"

        if not self.messages or self.messages[-1].get("role") != "assistant":
            self.messages.append({"role": "assistant", "content": assistant_text})
        self.last_bot_text = assistant_text
        self.tts.speak_async(assistant_text)

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower().replace("đ", "d")
        nfd = unicodedata.normalize("NFD", lowered)
        return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")

    def _drain_audio_queue(self, max_frames: int = 80) -> None:
        drained = 0
        while drained < max_frames:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
            drained += 1

    def _is_recent_duplicate(self, user_text: str, near_tts_window: bool, now: float) -> bool:
        if not self.last_user_text:
            return False
        age_s = now - self.last_user_at
        if age_s > 8.0:
            return False
        ratio = difflib.SequenceMatcher(
            None,
            self._normalize_text(user_text),
            self._normalize_text(self.last_user_text),
        ).ratio()
        if near_tts_window and age_s <= 5.0 and ratio >= 0.76:
            return True
        if age_s <= 3.0 and ratio >= 0.90:
            return True
        return False

    def _is_noise_like_text(self, user_text: str) -> bool:
        norm = self._normalize_text(user_text).strip()
        if not norm:
            return True
        tokens = [tok for tok in norm.split() if tok]
        if not tokens:
            return True
        if len(norm) >= 170:
            return True
        counts = Counter(tokens)
        max_repeat = max(counts.values())
        unique_ratio = len(counts) / max(1, len(tokens))
        short_ratio = sum(1 for tok in tokens if len(tok) <= 2) / max(1, len(tokens))
        # Common ASR gibberish pattern from speaker leakage: repeated short syllables.
        if len(tokens) >= 8 and max_repeat >= 4 and unique_ratio <= 0.55:
            return True
        if len(tokens) >= 8 and short_ratio >= 0.6:
            return True
        return False

    def _is_likely_echo(self, user_text: str, in_post_tts_guard: bool = False) -> bool:
        if not self.last_bot_text:
            return False
        now = time.time()
        elapsed_from_start_ms = (now - self.tts.last_started_at) * 1000.0
        elapsed_from_end_ms = (
            (now - self.tts.last_ended_at) * 1000.0 if self.tts.last_ended_at > 0 else 99_999.0
        )
        near_tts_window = self.tts.is_speaking or elapsed_from_end_ms <= 1800.0
        if elapsed_from_start_ms > 12000 and not near_tts_window:
            return False

        norm_user = self._normalize_text(user_text).strip()
        norm_bot = self._normalize_text(self.last_bot_text).strip()
        if not norm_user or not norm_bot:
            return False

        if in_post_tts_guard and len(norm_user) >= 4 and (
            norm_user in norm_bot or norm_bot in norm_user
        ):
            return True

        if len(norm_user) < 8 or len(norm_bot) < 8:
            return False

        ratio = difflib.SequenceMatcher(None, norm_user, norm_bot).ratio()
        if ratio >= 0.55:
            return True

        user_tokens = {tok for tok in norm_user.split() if len(tok) >= 3}
        bot_tokens = {tok for tok in norm_bot.split() if len(tok) >= 3}
        if user_tokens and bot_tokens:
            inter = user_tokens & bot_tokens
            overlap = len(inter) / max(1, min(len(user_tokens), len(bot_tokens)))
            if overlap >= 0.45 and len(inter) >= 3 and near_tts_window:
                return True
            fuzzy_shared = 0
            for u_tok in user_tokens:
                if any(difflib.SequenceMatcher(None, u_tok, b_tok).ratio() >= 0.72 for b_tok in bot_tokens):
                    fuzzy_shared += 1
            if near_tts_window and fuzzy_shared >= 2:
                return True
            if in_post_tts_guard and fuzzy_shared >= 1:
                return True

        bot_keywords = {"tro", "ly", "vnpost", "telecom", "goi", "cuoc", "tu", "van", "sim"}
        shared_keywords = len((user_tokens & bot_keywords) & bot_tokens)
        if shared_keywords >= 3 and near_tts_window:
            return True
        if shared_keywords >= 2 and in_post_tts_guard:
            return True
        if near_tts_window and ratio >= 0.42:
            return True
        return False

    def _smalltalk_reply(self, user_text: str) -> str | None:
        norm = self._normalize_text(user_text)
        if any(k in norm for k in ("xin chao", "chao ban", "hello", "hi")) and len(norm) <= 45:
            return (
                "Chào anh/chị. Em có thể hỗ trợ báo giá gói cước, tư vấn gói phù hợp "
                "hoặc tra cứu thuê bao."
            )
        if any(k in norm for k in ("co the ho tro", "ho tro gi", "ban giup toi")) and len(norm) <= 55:
            return (
                "Em hỗ trợ 3 việc chính: báo giá gói cước, tư vấn gói theo nhu cầu "
                "và tra cứu thuê bao theo số điện thoại."
            )
        return None
