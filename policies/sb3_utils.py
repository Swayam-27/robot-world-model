"""Load Stable-Baselines3 policies safely on Kaggle."""
import sys
from pathlib import Path


def load_sac(path: str | Path):
    """Load a saved SAC model on CPU.

    SB3 -> torch.utils.tensorboard -> tensorboard imports TensorFlow if it is installed (Kaggle has it).
    TensorFlow ships its own LLVM, which clashes with MuJoCo's OpenGL renderers (OSMesa: "LLVM ERROR:
    inconsistency in registered CommandLine options"; EGL: segfault). We never use TensorFlow, so block it.
    """
    sys.modules["tensorflow"] = None
    from stable_baselines3 import SAC

    return SAC.load(str(path), device="cpu")
