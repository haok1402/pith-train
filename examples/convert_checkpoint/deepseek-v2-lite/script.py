from pathlib import Path

from huggingface_hub import snapshot_download

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path("workspace/checkpoints/deepseek-v2-lite/hf-import")
cfg.save_path = Path("workspace/checkpoints/deepseek-v2-lite/torch-dcp/00000000")

if __name__ == "__main__":
    snapshot_download(repo_id="deepseek-ai/DeepSeek-V2-Lite", local_dir=cfg.load_path)
    launch(cfg)

cfg = ConvertCheckpointCfg()
cfg.operation = "dcp2hf"
cfg.load_path = Path("workspace/checkpoints/deepseek-v2-lite/torch-dcp/00000000")
cfg.save_path = Path("workspace/checkpoints/deepseek-v2-lite/hf-export")

if __name__ == "__main__":
    launch(cfg)
