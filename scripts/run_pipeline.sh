#!/usr/bin/env bash
# NanoLM one-shot pipeline (bash equivalent of run_pipeline.ps1).
# Idempotent: stages are skipped when their outputs already exist.
# Usage: bash scripts/run_pipeline.sh [--force] [--skip-pretrain] [--skip-finetune] [--max-minutes 110]
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=0; SKIP_PRETRAIN=0; SKIP_FINETUNE=0; MAX_MINUTES=110
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-pretrain) SKIP_PRETRAIN=1 ;;
    --skip-finetune) SKIP_FINETUNE=1 ;;
    --max-minutes=*) MAX_MINUTES="${arg#*=}" ;;
  esac
done

echo "== [0/6] environment check =="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

echo "== [1/6] corpus =="
if [ -f data/corpus/corpus.txt ] && [ "$FORCE" = 0 ]; then
  echo "  corpus.txt exists, skipping"
else
  python scripts/build_corpus.py --no-download
fi

echo "== [2/6] tokenizer + data =="
if [ -f data/processed/tokenizer.json ] && [ -f data/processed/train.bin ] && [ "$FORCE" = 0 ]; then
  echo "  data/processed ready, skipping"
else
  python scripts/prepare_data.py --vocab-size 12000
fi

echo "== [3/6] pretraining (resume-aware, ${MAX_MINUTES} min budget) =="
if [ -f out/pretrain/best.ckpt ] && [ "$FORCE" = 0 ]; then
  echo "  out/pretrain/best.ckpt exists, skipping"
elif [ "$SKIP_PRETRAIN" = 0 ]; then
  python -m nanollm.train --data-dir data/processed --out-dir out/pretrain \
    --tokenizer data/processed/tokenizer.json --max-minutes "$MAX_MINUTES" \
    --batch-size 32 --block-size 256 --n-layer 7 --n-head 8 --n-embd 512 \
    --lr 6e-4 --warmup-steps 300 --eval-interval 300 --eval-iters 20 \
    --log-interval 50 --sample-interval 600
fi

echo "== [4/6] QA + LoRA fine-tuning =="
if [ ! -f data/qa/qa_train.jsonl ] || [ "$FORCE" = 1 ]; then python scripts/build_qa.py; fi
if [ -f out/lora/lora_best.pt ] && [ "$FORCE" = 0 ]; then
  echo "  out/lora/lora_best.pt exists, skipping"
elif [ "$SKIP_FINETUNE" = 0 ]; then
  python -m nanollm.finetune --base-ckpt out/pretrain/best.ckpt \
    --tokenizer data/processed/tokenizer.json --out-dir out/lora \
    --train-jsonl data/qa/qa_train.jsonl --val-jsonl data/qa/qa_val.jsonl \
    --batch-size 16 --block-size 256 --epochs 60 --max-minutes 55 \
    --r 8 --alpha 16 --lr 3e-4 --warmup-steps 20 --log-interval 10
fi

echo "== [5/6] merge =="
if [ -f checkpoints/merged.pt ] && [ "$FORCE" = 0 ]; then
  echo "  checkpoints/merged.pt exists, skipping"
else
  python -m nanollm.merge --base-ckpt out/pretrain/best.ckpt \
    --lora out/lora/lora_best.pt --out checkpoints/merged.pt
fi

echo "== [6/6] before/after comparison =="
python scripts/compare_samples.py

echo "== done =="
echo "serve:  python -m nanollm.server.app --model checkpoints/merged.pt --port 8000"
echo "chat:   python -m nanollm.chat --ckpt checkpoints/merged.pt"
