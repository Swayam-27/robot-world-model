# Learning resources

Study in this order. Nothing here builds this exact project; these are the pieces it is made from.

## 1. Diffusion basics (before Step 5)
- Hugging Face Diffusion Models Course, Unit 1 (free, implements diffusion from scratch): https://huggingface.co/learn/diffusion-course · notebooks: https://github.com/huggingface/diffusion-models-class
- Small from-scratch DDPM repos to read: https://github.com/harmya/simple-diffusion
- fast.ai Practical Deep Learning Part 2 (free videos, builds Stable Diffusion from scratch; long, pick lessons as needed): https://course.fast.ai/Lessons/part2.html

## 2. How a world model pipeline is structured
- PyTorch reimplementation of the original World Models paper (Ha & Schmidhuber, CarRacing: VAE + RNN + controller): https://github.com/ctallec/world-models

## 3. The codebase this project adapts (Step 5 onwards)
- DIAMOND (diffusion world model, NeurIPS 2024 spotlight): https://github.com/eloialonso/diamond
  - Run the pretrained Atari demo first: `python src/play.py --pretrained`
  - Paper: https://arxiv.org/abs/2405.12399

## 4. World models for robots (framing and README)
- 1X World Model Challenge starter code (humanoid robot world models): https://github.com/1x-technologies/1xgpt
- 1X World Model report: https://www.1x.tech/1x-world-model.pdf
- GigaWorld-1, world models as robot policy evaluators: https://arxiv.org/abs/2607.02642

## 5. Seeing MuJoCo and policies
- Gymnasium Pusher-v5 docs (observation and action layout): https://gymnasium.farama.org/environments/mujoco/pusher/
- Laptop viewer snippet: Step 1, item 0 in `BUILD_GUIDE.md`
- A pretrained Pusher policy to compare against: https://huggingface.co/moona-ai/mujoco-pusher-v5-ppo
- SAC explained (OpenAI Spinning Up; read the intro, skip the maths at first): https://spinningup.openai.com/en/latest/algorithms/sac.html
- Stable-Baselines3 SAC docs (parameters, saving/loading): https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

## 6. Phase 2: a real arm (after January)
- SO-101 open-source arm (Hugging Face, works with LeRobot): https://www.hackster.io/news/hugging-face-launches-the-so-101-an-upgraded-low-cost-3d-printable-autonomous-robot-arm-532360f441eb
- Unofficial India build guide (parts from Evelta and Robu.in, about ₹25,000 to ₹27,000 for leader + follower as of April 2026): https://prathamv0811.github.io/so101-india-guide/

## Compute notes
- Kaggle: about 30 free GPU hours a week (T4/P100, 16 GB), sessions up to 12 hours, 20 GB working disk.
- Kaggle private datasets: 100 GB per dataset and 100 GB total quota (uncompressed). This project needs about 3 GB.
