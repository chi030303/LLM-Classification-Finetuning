from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-14B",
    local_dir="/root/autodl-tmp/llm_classification_finetuning/base_models/Qwen3-14B",
    resume_download=True
)
