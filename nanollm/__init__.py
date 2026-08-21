"""NanoLM: a from-scratch GPT pretrained on a small domain corpus and
fine-tuned with self-implemented LoRA, served through a local FastAPI chat.

Submodules:
- tokenizer : self-implemented byte-level BPE (trained on the corpus)
- data      : token loading + batching (pretrain & instruction fine-tune)
- model     : GPT transformer built from scratch in PyTorch
- train     : pretraining loop (AdamW, cosine schedule, grad clip, ckpt)
- sample    : autoregressive sampling (temperature / top-k)
- lora      : low-rank adapters injected into attention/MLP layers
- finetune  : LoRA instruction fine-tuning loop
- chat      : interactive CLI chat
- server    : FastAPI + SSE streaming chat server with a web UI
"""

__version__ = "0.1.0"
