#!/usr/bin/env python3
"""CLI demo for VNPost telecom callbot."""

from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.callbot import TelecomCallbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run telecom callbot demo")
    parser.add_argument(
        "--mode",
        choices=["inbound", "outbound"],
        default="inbound",
        help="Conversation mode",
    )
    parser.add_argument(
        "--phone",
        default="0987000001",
        help="Phone number for subscriber lookup and lead/callback logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bot = TelecomCallbot(mode=args.mode, phone=args.phone)

    print(f"[BOT] {bot.opening()}")
    print("[INFO] Nhap 'exit' de ket thuc.")

    while True:
        user_text = input("[USER] ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("[BOT] Ket thuc cuoc goi demo. Cam on anh/chị.")
            break

        print(f"[BOT] {bot.reply(user_text)}")


if __name__ == "__main__":
    main()
