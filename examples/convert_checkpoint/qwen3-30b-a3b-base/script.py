from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path("workspace/checkpoints/qwen3-30b-a3b-base/hf-import")
cfg.save_path = Path("workspace/checkpoints/qwen3-30b-a3b-base/torch-dcp/00000000")

if __name__ == "__main__":
    snapshot_download(repo_id="Qwen/Qwen3-30B-A3B-Base", local_dir=cfg.load_path)
    launch(cfg)
