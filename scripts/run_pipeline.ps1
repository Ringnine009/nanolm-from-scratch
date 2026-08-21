# NanoLM one-shot pipeline: corpus -> tokenizer -> pretrain -> LoRA finetune
# -> merge -> before/after sampling comparison.
#
# Idempotent: each stage is skipped if its output already exists. Re-run with
# -Force to redo every stage, -SkipPretrain / -SkipFinetune to skip stages.
# Pretraining auto-resumes from out/pretrain/latest.ckpt if interrupted.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 [-Force]
param(
    [switch]$Force,
    [switch]$SkipPretrain,
    [switch]$SkipFinetune,
    [string]$MaxMinutes = "110"   # pretraining wall-clock budget
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== [0/6] environment check =="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "torch/CUDA check failed" }

Write-Host "`n== [1/6] corpus =="
if ((Test-Path data\corpus\corpus.txt) -and -not $Force) {
    Write-Host "  corpus.txt exists, skipping (re-run with -Force to rebuild)"
} else {
    python scripts\build_corpus.py --no-download
}

Write-Host "`n== [2/6] tokenizer + tokenized data =="
if ((Test-Path data\processed\tokenizer.json) -and (Test-Path data\processed\train.bin) -and -not $Force) {
    Write-Host "  data/processed is ready, skipping (re-run with -Force to redo)"
} else {
    python scripts\prepare_data.py --vocab-size 12000
}

Write-Host "`n== [3/6] pretraining (resumes from latest.ckpt; budget ${MaxMinutes} min) =="
if ((Test-Path out\pretrain\best.ckpt) -and -not $Force) {
    Write-Host "  out/pretrain/best.ckpt exists, skipping (re-run with -Force to retrain)"
} elseif (-not $SkipPretrain) {
    python -m nanollm.train --data-dir data/processed --out-dir out/pretrain `
        --tokenizer data/processed/tokenizer.json --max-minutes $MaxMinutes `
        --batch-size 32 --block-size 256 --n-layer 7 --n-head 8 --n-embd 512 `
        --lr 6e-4 --warmup-steps 300 --eval-interval 300 --eval-iters 20 `
        --log-interval 50 --sample-interval 600
}

Write-Host "`n== [4/6] QA set + LoRA fine-tuning =="
if (-not (Test-Path data\qa\qa_train.jsonl) -or $Force) { python scripts\build_qa.py }
if ((Test-Path out\lora\lora_best.pt) -and -not $Force) {
    Write-Host "  out/lora/lora_best.pt exists, skipping (re-run with -Force to refinetune)"
} elseif (-not $SkipFinetune) {
    python -m nanollm.finetune --base-ckpt out\pretrain\best.ckpt `
        --tokenizer data/processed/tokenizer.json --out-dir out/lora `
        --train-jsonl data/qa/qa_train.jsonl --val-jsonl data/qa/qa_val.jsonl `
        --batch-size 16 --block-size 256 --epochs 60 --max-minutes 55 `
        --r 8 --alpha 16 --lr 3e-4 --warmup-steps 20 --log-interval 10
}

Write-Host "`n== [5/6] merge LoRA into base =="
if ((Test-Path checkpoints\merged.pt) -and -not $Force) {
    Write-Host "  checkpoints/merged.pt exists, skipping (re-run with -Force to remerge)"
} else {
    python -m nanollm.merge --base-ckpt out\pretrain\best.ckpt `
        --lora out\lora\lora_best.pt --out checkpoints\merged.pt
}

Write-Host "`n== [6/6] before/after sampling comparison =="
python scripts\compare_samples.py --base out\pretrain\best.ckpt --merged checkpoints\merged.pt

Write-Host "`n== done =="
Write-Host "serve:  python -m nanollm.server.app --model checkpoints\merged.pt --port 8000"
Write-Host "chat:   python -m nanollm.chat --ckpt checkpoints\merged.pt"
