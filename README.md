# AI-callbot-Demo (Telecom Scope)

Demo callbot AI cho VNPost Telecom, chỉ tập trung nghiệp vụ viễn thông:
- Mời mua SIM (outbound telesales).
- Tra cứu thông tin gói cước (inbound support).
- Tư vấn gói cước phù hợp dựa trên nhu cầu.

Repo này là bản PoC sử dụng `FastAPI` + `sample data` + `rule-based dialogue`.

## 1) Tính năng demo

### Inbound
- Tra cứu gói cước đang dùng theo số thuê bao.
- Tư vấn gói cước mới theo nhu cầu data/call/chi phí.
- Chuyển tổng đài viên khi khách yêu cầu.

### Outbound
- Bot gọi mời mua SIM khuyến mãi.
- Xử lý nhánh `đồng ý` -> tạo lead bán hàng.
- Xử lý nhánh `từ chối` -> đặt lịch gọi lại.

## 2) Kiến trúc PoC

- `app/main.py`: API endpoint phục vụ callbot.
- `app/sample_data.py`: dữ liệu mẫu gói cước, SIM, thuê bao.
- `app/service.py`: logic nghiệp vụ (recommend, lead, callback).
- `app/callbot.py`: state machine hội thoại demo inbound/outbound.
- `scripts/demo_conversation.py`: chạy hội thoại demo trên terminal.

## 3) Cài đặt và chạy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập docs:
- Swagger UI: `http://localhost:8000/docs`

### Chạy demo hội thoại

```bash
python scripts/demo_conversation.py --mode inbound --phone 0987000001
python scripts/demo_conversation.py --mode outbound --phone 0987000002
```

## 4) API chính

- `GET /health`
- `GET /telecom/plans`
- `GET /telecom/plans/{plan_id}`
- `GET /telecom/sim-offers`
- `GET /telecom/subscribers/{phone}`
- `POST /telecom/recommend-plan`
- `POST /telecom/leads`
- `POST /telecom/callbacks`

## 5) Kịch bản demo gợi ý (cho buổi trình bày)

### Inbound script
1. Khách: "Tra cuu goi dang dung."
2. Bot: Trả gói hiện tại + mức dùng data.
3. Khách: "Tu van goi cuoc phu hop."
4. Bot: Đề xuất gói mới, nêu lý do.
5. Khách: "Toi muon gap tong dai vien."
6. Bot: Xác nhận chuyển máy.

### Outbound script
1. Bot: Chào và giới thiệu ưu đãi SIM.
2. Khách: "Dong y, tu van nhanh."
3. Bot: Đưa 2 lựa chọn SIM.
4. Khách: Chọn một gói.
5. Bot: Tạo lead và xác nhận thời gian nhân viên gọi lại.

### Host model AI
ct2-transformers-converter \
  --model vinai/PhoWhisper-small \
  --output_dir models/PhoWhisper-small-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8_float16

vllm serve Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --host 0.0.0.0 --port 8000 \
  --dtype half \
  --max-model-len 2048 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90

## 6) Realtime pipeline (PhoWhisper + vLLM + barge-in)

Mục tiêu:
- ASR: `vinai/PhoWhisper-small` đã convert sang CTranslate2.
- LLM: `Qwen/Qwen3-1.7B-GPTQ-Int8` qua vLLM.
- Hành vi latency thấp cho callbot: `enable_thinking=false`.
- Hỗ trợ `barge-in`: khách chen lời thì bot dừng phát TTS ngay.
- Tích hợp `tool-system` để dùng sample data (tra cứu thuê bao, gợi ý gói, ưu đãi SIM...).

### Cài dependency cho realtime

```bash
pip install -r requirements-realtime.txt
```

### Convert PhoWhisper sang CTranslate2

```bash
./scripts/convert_phowhisper_ct2.sh
```

Hoặc command tay:

```bash
ct2-transformers-converter \
  --model vinai/PhoWhisper-small \
  --output_dir models/PhoWhisper-small-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8_float16
```

### Chạy vLLM server (VRAM 6GB khuyến nghị)

```bash
vllm serve Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --host 0.0.0.0 --port 8002 \
  --dtype half \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.75
```

### Chạy realtime callbot

```bash
python scripts/realtime_callbot.py \
  --vllm-base-url http://localhost:8002/v1 \
  --asr-model-path models/PhoWhisper-small-ct2 \
  --tts-mode text \
  --vad-aggressiveness 3 \
  --min-speech-ms 260 \
  --endpoint-silence-ms 550 \
  --barge-in-ms 320 \
  --utterance-min-rms 0.012

```

Nếu muốn phát âm thanh local:

```bash
python scripts/realtime_callbot.py --tts-mode pyttsx3
```

Nếu muốn dùng VieNeu-TTS (khuyến nghị bản nhẹ `0.3B-q4-gguf`):

```bash
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/
python scripts/realtime_callbot.py \
  --tts-mode vieneu \
  --tts-vieneu-backbone-repo pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf \
  --tts-vieneu-backbone-device cpu
```

Mặc định VieNeu chạy non-stream (ổn định hơn, giảm lỗi ALSA underrun).  
Nếu muốn ưu tiên độ trễ thấp, bật streaming:

```bash
python scripts/realtime_callbot.py \
  --tts-mode vieneu \
  --tts-vieneu-backbone-repo pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf \
  --tts-vieneu-backbone-device cpu \
  --tts-vieneu-streaming
```

Nếu micro vẫn ăn phải tiếng loa, tăng độ ổn định bằng:

```bash
python scripts/realtime_callbot.py \
  --tts-mode vieneu \
  --tts-vieneu-backbone-repo pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf \
  --tts-vieneu-backbone-device cpu \
  --barge-in-ms 420 \
  --barge-in-min-rms 0.035 \
  --utterance-min-rms 0.015 \
  --post-tts-guard-ms 1400 \
  --post-tts-guard-rms 0.035
```

Để chọn đúng mic/loa (tránh loopback thiết bị mặc định):

```bash
python scripts/realtime_callbot.py --list-audio-devices
python scripts/realtime_callbot.py \
  --tts-mode vieneu \
  --input-device 2 \
  --output-device 5
```

Note:
- `barge-in` mặc định 320ms speech liên tục để ngắt bot (`--barge-in-ms`).
- `enable_thinking=false` đã được hard-code trong `app/realtime_pipeline.py` khi gọi vLLM OpenAI API.
- Để tránh echo làm false-barge-in khi demo local, nên dùng tai nghe.
- Nếu môi trường nhiều tạp âm thanh, thử `--disable-barge-in` hoặc tăng `--barge-in-ms`.
- Nếu ASR nhảy câu vô nghĩa, tăng ngưỡng `--utterance-min-rms` lên `0.012` -> `0.02`.

## 7) Gradio UI (Kaggle friendly)

File chạy web UI: `app_gradio.py`

Tính năng:
- Text chat với vLLM (OpenAI-compatible API).
- Voice input (microphone/upload) -> ASR PhoWhisper -> gửi vào chatbot.
- Realtime voice trong browser: bật `Bật realtime voice`, bot tự gửi khi bạn dừng nói.
- Có fallback rule-based nếu vLLM chưa sẵn sàng.
- Request tới vLLM có `enable_thinking=false` để giảm độ trễ.
- Có tool-system dùng `sample_data` để bot trả lời logic hơn cho các tác vụ:
  tra cứu thuê bao, danh sách gói cước, ưu đãi SIM, gợi ý gói, tạo lead và lịch gọi lại.

### Chạy local

```bash
pip install -r requirements-gradio.txt
python app_gradio.py --host 0.0.0.0 --port 7860
```

Nếu vLLM đang chạy cổng 8002:
- Set `vLLM Base URL` trong UI: `http://localhost:8002/v1`

### Chạy trên Kaggle notebook

```python
!git clone https://github.com/VuThanhLam124/AI-callbot-Demo.git
%cd AI-callbot-Demo
!pip install -r requirements-gradio.txt
!python app_gradio.py --host 0.0.0.0 --port 7860
```

Nếu bạn chưa dựng vLLM trên Kaggle, bỏ check `Dùng vLLM` trong UI để test fallback flow.
