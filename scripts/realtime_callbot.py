#!/usr/bin/env python3
"""Run realtime telecom callbot with PhoWhisper + vLLM."""

from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.realtime_pipeline import (
    AudioConfig,
    FasterWhisperASR,
    RealtimeCallbot,
    TTSPlayer,
    VLLMChatClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime callbot demo for VNPost Telecom")

    parser.add_argument(
        "--asr-model-path",
        default="models/PhoWhisper-small-ct2",
        help="Path to converted CTranslate2 ASR model directory",
    )
    parser.add_argument("--asr-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--asr-compute-type", default="int8_float16")
    parser.add_argument("--asr-language", default="vi")
    parser.add_argument("--asr-beam-size", type=int, default=2)
    parser.add_argument("--asr-no-speech-threshold", type=float, default=0.6)
    parser.add_argument("--asr-log-prob-threshold", type=float, default=-1.0)

    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-1.7B-GPTQ-Int8")
    parser.add_argument("--vllm-api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=160)

    parser.add_argument("--tts-mode", default="text", choices=["text", "pyttsx3"])
    parser.add_argument("--tts-rate", type=int, default=180)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--vad-aggressiveness", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--min-speech-ms", type=int, default=260)
    parser.add_argument("--endpoint-silence-ms", type=int, default=550)
    parser.add_argument("--utterance-min-rms", type=float, default=0.010)
    parser.add_argument("--barge-in-min-rms", type=float, default=0.015)
    parser.add_argument(
        "--barge-in-ms",
        type=int,
        default=320,
        help="Continuous speech duration needed to interrupt bot speech",
    )
    parser.add_argument("--barge-in-min-bot-ms", type=int, default=250)
    parser.add_argument(
        "--disable-barge-in",
        action="store_true",
        help="Disable interruption while bot is speaking",
    )

    parser.add_argument(
        "--system-prompt",
        default=(
            "You are VNPost Telecom voice assistant. "
            "Answer in concise Vietnamese. "
            "Focus on SIM offers and telecom package lookup. "
            "If user asks outside telecom scope, politely redirect."
        ),
    )
    parser.add_argument(
        "--greeting",
        default=(
            "Xin chao, day la tro ly AI VNPost Telecom. "
            "Toi co the ho tro tra cuu goi cuoc hoac tu van mua SIM."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = AudioConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        vad_aggressiveness=args.vad_aggressiveness,
        min_speech_ms=args.min_speech_ms,
        endpoint_silence_ms=args.endpoint_silence_ms,
        utterance_min_rms=args.utterance_min_rms,
        barge_in_min_rms=args.barge_in_min_rms,
    )
    barge_in_frames = max(1, args.barge_in_ms // args.frame_ms)

    asr = FasterWhisperASR(
        model_path=args.asr_model_path,
        device=args.asr_device,
        compute_type=args.asr_compute_type,
        language=args.asr_language,
        beam_size=args.asr_beam_size,
        no_speech_threshold=args.asr_no_speech_threshold,
        log_prob_threshold=args.asr_log_prob_threshold,
    )
    llm = VLLMChatClient(
        base_url=args.vllm_base_url,
        model=args.vllm_model,
        api_key=args.vllm_api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    tts = TTSPlayer(mode=args.tts_mode, rate_wpm=args.tts_rate)

    bot = RealtimeCallbot(
        asr=asr,
        llm=llm,
        tts=tts,
        audio_cfg=cfg,
        system_prompt=args.system_prompt,
        greeting=args.greeting,
        barge_in_frames=barge_in_frames,
        barge_in_min_bot_ms=args.barge_in_min_bot_ms,
        barge_in_enabled=not args.disable_barge_in,
    )
    bot.run()


if __name__ == "__main__":
    main()
