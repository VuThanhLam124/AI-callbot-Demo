#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-vinai/PhoWhisper-small}"
OUTPUT_DIR="${2:-models/PhoWhisper-small-ct2}"
QUANTIZATION="${3:-int8_float16}"

echo "[INFO] Converting ${MODEL_ID} -> ${OUTPUT_DIR} (quantization=${QUANTIZATION})"

ct2-transformers-converter \
  --model "${MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization "${QUANTIZATION}"

echo "[DONE] Converted model is ready at: ${OUTPUT_DIR}"
