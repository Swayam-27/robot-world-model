# The project, explained like a story

Read this on the bus. Each idea comes in three parts: the everyday picture, the real term, and where it lives in the repo. `CONCEPTS.md` is the dictionary; this file is the story. A new chapter is added after every step.

---

## The pitch in 20 seconds

A driving school bought a driving simulator. Before trusting it, they check one thing: **do students who drive well in the simulator also drive well on the real road?** If the simulator ranks the students in the same order as the road test, it is a trustworthy simulator.

This project does that for a robot arm:
- **The road** is MuJoCo, a physics simulator.
- **The students** are about 10 robot brains of different skill.
- **The simulator under test** is a neural network that learned to imagine the arm's world just by watching videos of it. That network is the **world model**, and the video it imagines is the **dream**.

The final grade is one number: **how well the dream's ranking of the brains matches the real ranking.**

---

## The cast of characters

| Everyday picture | Real term | Where in the repo |
| --- | --- | --- |
| The real world / the road test | **MuJoCo Pusher-v5**, the environment | `envs/pusher.py` |
| One attempt at the task | **Episode** (100 steps) | |
| One tick of the clock | **Step** | |
| What the robot "feels" each tick | **State / observation**: 23 numbers | `envs/constants.py` |
| What the robot does each tick | **Action**: 7 joint twists (torques) | |
| A photo from the fixed camera | **Frame**, 64x64 pixels | `PusherEnv.frame()` |
| The robot's brain | **Policy** | `policies/` |
| The coach that trains the brain | **SAC**, a reinforcement learning algorithm | `policies/train_sac.py` |
| Treats and scoldings while practising | **Reward** | inside Pusher |
| Photos of the student at every grade | **Checkpoints** | `outputs/.../ckpts/` |
| Practice questions | **Training seeds** 0 to 9,999 | `TRAIN_SEEDS` |
| The final exam, kept sealed | **Eval seeds** 10,000 to 10,049 | `EVAL_SEEDS` |
| The robot's eyes inside the dream | **State reader**, a small CNN | `models/` (Step 4) |
| The imagination | **World model**, a diffusion network | `models/` (Step 5) |
| The report cards | **Rankings A, B, C** | `eval/` |
| How well two report cards agree | **Spearman correlation** | Step 8 |

**Watch out for one mix-up:** a **policy** is the *brain*, the thing that decides moves. **Changing one setting in a test** (like training for longer) is an **experiment**, and each experiment is one config file in `configs/`. Brains are what we rank; experiments are how we build and improve things.

---

## Chapter 1 · Building the stage (Step 1, done)

**The task.** A 7-joint arm (shoulder, elbow, wrist...) sits at a table. There's a can and a goal spot. The arm must push the can onto the spot within 100 ticks. Every tick, the brain reads 23 numbers (joint angles, joint speeds, and where the fingertip, can and goal are) and answers with 7 numbers (how hard to twist each joint).

**Seeds = shuffling the deck.** Each attempt starts with the can and goal in a random place. A **seed** is a number that fixes that "random" choice, so seed 42 always gives exactly the same start. That makes tests repeatable, and it lets us keep a sealed exam: start positions no brain or model ever practises on.

**The camera.** The world model will only ever see pictures, so we fixed one camera angle that keeps the arm, can and goal in view (`CAMERA_CONFIG`). Pictures are tiny (64x64) on purpose: the dream must draw every pixel, 100 times per episode, on a free GPU.

**The plug socket.** All code talks to the world through four calls: `reset(seed)`, `step(action)`, `frame()`, `state()` (`envs/interface.py`). Today MuJoCo is behind the socket. In phase 2 a real arm and a webcam could plug into the same socket, and nothing else changes.

---

## Chapter 2 · Teaching the brains (Step 2, done)

### Reinforcement learning = training a dog with treats
Nobody shows the arm how to push. It tries things, gets a score, and slowly does more of what scored well. That's **reinforcement learning (RL)**: learning by trial and error from a score called the **reward**.

### The reward, and what "reward weights" means
Every tick, Pusher hands out a reward made of three parts. Each part has a **weight**, a number that says how much that part matters:

```
reward = − 1.0 × (can-to-goal distance)       "get the can to the goal"   ← the real job
         − 0.5 × (fingertip-to-can distance)  "go near the can"           ← a hint
         − 0.1 × (how hard the joints twist)  "don't waste energy"        ← a small fine
```

The reward is always negative, like a penalty meter, and closer to 0 is better. **The weights 1.0, 0.5 and 0.1 are the reward weights.** They're like a teacher's marking scheme: if neatness counted for 50% of the grade, students would write very neatly and solve fewer problems. Change the weights and the brain learns a different habit. For example:
- a bigger **can-to-goal** weight makes the brain care more about finishing the push
- a smaller **energy** fine makes it bolder

That's why changing the reward weights is our *backup* fix: we'd only try it if longer training stops improving.

**Trap we avoided:** the reward isn't the score we rank brains by. A brain can earn a better reward just by moving less (a smaller energy fine) while the can never moves. So we judge every brain by one honest number: **how far the can ends from the goal** (`object_goal_distance`).

### SAC = a player and a coach
**SAC (Soft Actor-Critic)** is the training method. It has two parts:
- **The actor** is the player, the policy itself. It looks at the 23 numbers and picks the 7 twists.
- **The critics** are the coaches. They guess "from here, doing that, how much reward will you collect in the future?" The actor learns to pick moves the coaches like. SAC has *two* coaches and listens to the more pessimistic one, so it doesn't get over-confident.

Three more pieces:
- **Replay buffer = a diary.** Every tick is written down (what it saw, what it did, what it got). Training reads random pages from the whole diary, not just today's. Learning from old pages is called **off-policy**, and it's why SAC is efficient enough to run on a laptop-class CPU.
- **Entropy bonus = a curiosity reward.** SAC gets a little extra reward for staying slightly random, so it keeps trying new moves instead of locking into the first okay trick. That's the "Soft" in the name.
- **Deterministic = no dice when grading.** During training the brain adds a bit of randomness. During testing we switch it off (`deterministic=True`), so the same brain on the same seed always does exactly the same thing.

### Checkpoints = a skill ladder for free
While training, we save a photo of the brain every 20,000 steps (**checkpoints**). Early photos are clumsy, later ones are skilled. That's exactly what we need: the exam must be taken by students of *different* levels, otherwise any ranking is just noise.

### What our first run showed (300k steps, measured on 20 practice seeds)
| Checkpoint | Can ends this far from goal | Can moved at all? |
| --- | --- | --- |
| 20k | 0.258 m (start was ~0.25 m) | 5% of tries |
| 100k | 0.229 m | 25% |
| 200k | 0.177 m | 75% |
| 300k | **0.115 m** | 100% |

The brain *does* learn to push, and it was still getting better when training stopped. But it leaves the can about 11 cm short of the goal. So the fix is the simplest one: **practise longer**. That's a 1M-step run that changes only `total_timesteps` (`configs/sac_pusher_1m.yaml`). The first run wasn't wasted: its checkpoints are a ready-made ladder of weaker students.

**Why not a GPU?** The brain's network is tiny and the physics runs on the CPU, so a GPU would sit idle. We save GPU hours for the world model.

### The class of 10 brains (chosen 2026-11-06)
We held **tryouts** on 50 practice seeds and kept 10 brains that are clearly different from each other:

```
best  sac_960k 0.054 m · sac_600k 0.074 · sac_400k 0.097 · sac_260k 0.119
      960k + shaky hands 0.6 → 0.139 · 0.8 → 0.155 · 1.0 → 0.178 · 1.3 → 0.205
worst sac_100k 0.220 · random 0.241
```

- **Shaky hands** = the best brain with random jitter added to every move. More jitter, worse pushing. It's like the same student taking the exam with cold hands.
- **Near-tie check** = we compare two brains *on the same 50 seeds*, seed by seed (a **paired comparison**). If the average gap is smaller than the natural wobble (about 2 **standard errors**), we can't tell them apart, so one gets dropped. The 20k brain was dropped this way: it's just as useless as random.
- **Why not the last checkpoint?** The 1M brain was *worse* than the 960k one. RL brains can get a bit worse late in training, so we pick by measurement, not by age.

You can watch any of them live: `.venv\Scripts\python.exe -m policies.watch_live --name sac_960k`

### Spearman: the one number the whole project is judged by
Two report cards rank the same 10 students. **Spearman** asks: *do they put the students in the same order?*

1. Replace each score by its **rank** (1st, 2nd, ... 10th) on each report card.
2. For each student, take the rank difference *d* (2nd on one card, 4th on the other, so *d* = 2).
3. ρ = 1 − 6·Σd² / (n·(n² − 1)), with n = 10 students.

- ρ = **1**: identical order. ρ = **0**: no relation, like shuffling. ρ = **−1**: exactly reversed.
- **Why ranks, not raw scores?** The dream might be harsher on everyone (every brain looks 2 cm worse). That's fine. A team choosing "which brain do we deploy?" only needs the *order* to be right. Spearman ignores the harshness and checks only the order.
- **Our headline:** Spearman between ranking A (real MuJoCo) and ranking C (the dream).

### The sealed exam: ranking A (2026-11-06)
All 10 brains played the 50 eval seeds they had never seen (`policies/score.py`):
- **Same order as in practice.** Spearman(practice order, exam order) = **1.0**, so the tryouts weren't a fluke.
- **Played twice, identical to the last digit.** Nothing in our setup rolls unseeded dice.
- **Still no near-ties.** It runs from 0.058 m (`sac_960k`) to 0.245 m (random).

`outputs/scores/real_scores.csv` is now the **answer key**. Everything from Step 3 on tries to reproduce this order *without* the real simulator.

### The pass mark: 3 cm (2026-11-06)
We looked at 300 final pictures from practice episodes, sorted by how far the can ended from the goal:
- at **2 cm** the can sits right in the middle of the red disc
- at **4 cm** it has slid toward the rim

So **3 cm** is the line: closer than 3 cm counts as **success**, otherwise it's a miss. Each brain's **success rate** is the share of the 50 exam episodes it passes.

- **On the exam,** only the two best brains ever pass: `sac_960k` 30% and `sac_600k` 24%. Every other brain gets 0%. It's a strict exam.
- **The rule we kept:** we chose 3 cm from the pictures *before* seeing any success rates. Choosing the pass mark that makes the results look nicest would be cheating on the exam.
- **The ranking itself still uses distance,** which separates all 10 brains. Success rate is a second, easy-to-read number.

**Step 2 is done.** We have the students (10 brains), the answer key (ranking A) and the pass mark.

---

## The lab rules, and why they exist

| Rule | Everyday reason |
| --- | --- |
| Eval seeds never appear in training or tuning | No peeking at the exam paper, or a high score means nothing |
| Change one thing at a time | If you change two things and it improves, you don't know which one helped |
| Split data by episode, not by frame | Neighbouring frames are near-copies; splitting them is like testing on homework answers |
| Score with `deterministic=True` | Same student, same question, same answer, so rankings repeat |
| Never fake results | The whole point is a number you can defend in an interview |

---

## Coming chapters (one line each)

- **Step 3 · Record the dataset:** film every brain (and random flailing) for about 2,000 episodes. The world model only learns what it has seen, mistakes included.
- **Step 4 · The eyes:** a CNN learns to read the 23 numbers from pictures, because inside the dream there are only pictures.
- **Step 5 · The imagination:** a diffusion model learns "given the last 4 frames and the action, draw the next frame".
- **Step 6 · Dream for 100 steps:** feed its own drawings back in. The fight here is against **drift** (small errors snowballing).
- **Step 7 · Brains play in the dream:** dream frame, then eyes, then brain, then action, then the next dream frame. That gives **ranking C**.
- **Step 8 · The verdict:** does ranking C match ranking A? (Spearman)
- **Step 9 · Fix one thing** and measure again.
- **Step 10 · Ship:** demo, README, resume.
