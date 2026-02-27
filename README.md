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

Muc tieu:
- ASR: `vinai/PhoWhisper-small` da convert sang CTranslate2.
- LLM: `Qwen/Qwen3-1.7B-GPTQ-Int8` qua vLLM.
- Hanh vi latency-thap cho callbot: `enable_thinking=false`.
- Ho tro `barge-in`: khach chen loi thi bot dung phat TTS ngay.

### Cai dependency cho realtime

```bash
pip install -r requirements-realtime.txt
```

### Convert PhoWhisper sang CTranslate2

```bash
./scripts/convert_phowhisper_ct2.sh
```

Hoac command tay:

```bash
ct2-transformers-converter \
  --model vinai/PhoWhisper-small \
  --output_dir models/PhoWhisper-small-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8_float16
```

### Chay vLLM server

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

### Chay realtime callbot

```bash
python scripts/realtime_callbot.py \
  --asr-model-path models/PhoWhisper-small-ct2 \
  --vllm-base-url http://localhost:8002/v1 \
  --vllm-model Qwen/Qwen3-1.7B-GPTQ-Int8 \
  --tts-mode text \
  --vad-aggressiveness 3 \
  --min-speech-ms 260 \
  --endpoint-silence-ms 550 \
  --barge-in-ms 320
```

Neu muon phat am thanh local:

```bash
python scripts/realtime_callbot.py --tts-mode pyttsx3
```

Note:
- `barge-in` mac dinh 320ms speech lien tuc de ngat bot (`--barge-in-ms`).
- `enable_thinking=false` da duoc hard-code trong `app/realtime_pipeline.py` khi goi vLLM OpenAI API.
- De tranh echo lam false-barge-in khi demo local, nen dung tai nghe.
- Neu moi truong nhieu tap am thanh, thu `--disable-barge-in` hoac tang `--barge-in-ms`.
- Neu ASR nhay cau vo nghia, tang nguong `--utterance-min-rms` len `0.012` -> `0.02`.

## 7) Gradio UI (Kaggle friendly)

File chay web UI: `app_gradio.py`

Tinh nang:
- Text chat voi vLLM (OpenAI-compatible API).
- Voice input (microphone/upload) -> ASR PhoWhisper -> send vao chatbot.
- Co fallback rule-based neu vLLM chua san sang.
- Request toi vLLM co `enable_thinking=false` de giam do tre.

### Chay local

```bash
pip install -r requirements-gradio.txt
python app_gradio.py --host 0.0.0.0 --port 7860
```

Neu vLLM dang chay cong 8002:
- Set `vLLM Base URL` trong UI: `http://localhost:8002/v1`

### Chay tren Kaggle notebook

```python
!git clone https://github.com/VuThanhLam124/AI-callbot-Demo.git
%cd AI-callbot-Demo
!pip install -r requirements-gradio.txt
!python app_gradio.py --host 0.0.0.0 --port 7860
```

Neu ban chua dung vLLM tren Kaggle, bo check `Use vLLM` trong UI de test fallback flow.
