"""Load the unchanged split script lazily; missing scripts fail, never skip."""
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "split_dataset.py"
_cached_module = None


def load_split_module():
    global _cached_module
    if _cached_module is not None:
        return _cached_module
    if not SCRIPT.is_file():
        raise FileNotFoundError(
            f"The original split tests require {SCRIPT}. "
            "Keep scripts/split_dataset.py from your project; it is not replaced by this migration."
        )
    name = "split_dataset_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load split script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _cached_module = module
    return module
