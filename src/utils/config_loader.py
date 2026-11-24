from pathlib import Path
from typing import Any, Dict
import yaml
def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    Parameters
    path : str or Path: Path to YAML config file.
    Returns
    dict: Parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
