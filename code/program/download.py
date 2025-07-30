from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="CardinalOperations/ORLM-LLaMA-3-8B", local_dir="./ORLM-LLaMA-3-8B")
print(f"模型已下载至: {local_dir}")
