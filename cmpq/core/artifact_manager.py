import os
from pathlib import Path
from typing import Optional

class ArtifactManager:
    """
    Manages the file paths for output artifacts in the CMP-Q pipeline.
    """
    def __init__(self, output_dir: str, experiment_name: str):
        self.root = Path(output_dir) / experiment_name
        self.models_dir = self.root / "models"
        self.logs_dir = self.root / "logs"
        
        # Ensure directories exist
        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    @property
    def compressed_model_path(self) -> Path:
        """Directory where the final compressed model is saved"""
        return self.models_dir / "compressed_model"

    def get_log_file(self, stage_name: str) -> Path:
        return self.logs_dir / f"{stage_name}.log"
