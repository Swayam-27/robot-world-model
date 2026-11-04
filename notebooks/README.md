# Kaggle notebooks

Notebooks stay thin: they install, clone the repo and call scripts. All logic lives in `.py` files.

First cell of every notebook:

```python
!git clone https://github.com/<your-username>/robot-world-model.git
%cd robot-world-model
!pip install -q -r requirements.txt

import os
os.environ["MUJOCO_GL"] = "egl"   # fallback: "osmesa"
```

For a private repo, store a GitHub token in Kaggle Secrets and clone with it.

Tips:
- CPU sessions for the simulator, SAC and data recording; GPU only for training the reader and world model.
- Write checkpoints to `/kaggle/working/` every 20 to 30 minutes and save important ones as a Kaggle dataset version.
