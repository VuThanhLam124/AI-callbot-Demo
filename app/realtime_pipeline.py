"""Realtime callbot pipeline with VAD, ASR, LLM and barge-in support."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


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

    def __init__(self, mode: str = "text", rate_wpm: int = 180):
        self.mode = mode
        self.rate_wpm = rate_wpm
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._engine: Any = None
        self._last_started_at = 0.0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def last_started_at(self) -> float:
        return self._last_started_at

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

    def _speak_worker(self, text: str) -> None:
        self._is_speaking = True
        try:
            try:
                if self.mode == "pyttsx3":
                    self._speak_pyttsx3(text)
                else:
                    self._speak_text_mode(text)
            except Exception as exc:
                print(f"[WARN] TTS backend failed ({type(exc).__name__}). Fallback to text mode.")
                if not self._stop_event.is_set():
                    self._speak_text_mode(text)
        finally:
            self._is_speaking = False
            with self._lock:
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
        barge_in_frames: int = 8,
        barge_in_min_bot_ms: int = 250,
        barge_in_enabled: bool = True,
    ):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.audio_cfg = audio_cfg
        self.vad = StreamingVAD(audio_cfg)
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.barge_in_frames = max(1, barge_in_frames)
        self.barge_in_min_bot_ms = max(0, barge_in_min_bot_ms)
        self.barge_in_enabled = barge_in_enabled
        self.speech_during_tts = 0
        self.stop_event = threading.Event()
        self.audio_q: queue.Queue[bytes] = queue.Queue(maxsize=300)
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

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
        self.tts.speak_async(self.greeting)

        try:
            with sd.RawInputStream(
                samplerate=self.audio_cfg.sample_rate,
                blocksize=frame_samples,
                channels=1,
                dtype="int16",
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
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
        finally:
            self.tts.stop()

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
            print("[BARGE-IN] User interrupted bot speech.")
            self.tts.stop()
            self.speech_during_tts = 0

    def _handle_utterance(self, pcm16_audio: bytes) -> None:
        utterance_rms = _pcm16_rms(pcm16_audio)
        if utterance_rms < self.audio_cfg.utterance_min_rms:
            return

        user_text = self.asr.transcribe(pcm16_audio).strip()
        if not user_text or len(user_text) < 2:
            return
        print(f"[USER] {user_text}")
        self.messages.append({"role": "user", "content": user_text})

        try:
            assistant_text = self.llm.chat(self.messages).strip()
        except Exception as exc:
            assistant_text = (
                "Xin loi, toi dang gap loi ket noi AI server. "
                f"Chi tiet: {type(exc).__name__}"
            )

        if not assistant_text:
            assistant_text = "Xin loi, toi chua nghe ro. Ban co the noi lai giup toi khong?"

        self.messages.append({"role": "assistant", "content": assistant_text})
        self.tts.speak_async(assistant_text)
