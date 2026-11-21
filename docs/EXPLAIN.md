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

## Chapter 3 · Filming the brains (Step 3)

**Why film at all?** The world model learns only from videos, so it can only imagine what it has seen. If it only watched the best brain, it would never learn what a *miss* looks like, and it would make bad brains look good in the dream. So we filmed **every kind of behaviour**.

**What we filmed (`data/record.py`):**
- 2,000 episodes, each from a **different practice start position**
- 7 brains × 200 episodes, plus the **random brain × 600** (its flailing shows the world model moves no trained brain ever makes)
- Two brains, **`sac_400k` and `sac_960k_n0.8`, were never filmed.** In Step 8 we'll ask: can the dream judge brains it has *never* seen? That's the "generalisation" test.

**What one episode holds**, like a flipbook with notes:
- **101 frames**: pictures, 64×64
- **100 actions**: the twists the brain chose
- **101 states**: the true 23 numbers behind each picture

Frame *t* plus action *t* gives frame *t+1*. That's why there's one more frame than actions, and it's exactly the rule the world model must learn.

**Training and validation split by episode.** 90% of each brain's episodes are for training and 10% for validation (checking). We split whole episodes, never single frames. Neighbouring frames are near-copies, so splitting by frame would be like testing a student on the homework they just did.

**How we know the data is right: the replay check (`data/playback.py`).** For 2 episodes of every brain, we reset the simulator to the episode's start and replay the recorded actions. Every state and every frame came out **identical** to the file. If anything were shifted by even one step, the check would fail.

**A real-life hiccup.** The laptop ran low on memory twice, and the recording was stopped. Because each episode is its own file, and saving is "write fully, then rename", we simply restarted and nothing was lost.

## Chapter 4 · Giving the brains eyes (Step 4, in progress)

**Why eyes?** Our brains were trained reading the 23 true numbers straight from the simulator, like a player who is told exactly where everything is. Inside the dream there are no numbers, only pictures. So we need **eyes**: a network that looks at pictures and writes down the 23 numbers. That's the **state reader** (`models/reader.py`).

**The warm-up first (`warmup/faces_cnn_torch.py`).** Before building the eyes, we rebuilt the old faces CNN (real vs AI faces, from Keras) in PyTorch. It was a practice run for the PyTorch training loop: `zero_grad -> forward -> loss -> backward -> step`. It reached 85.8% validation accuracy.

**How the eyes are built.** We used the same conv blocks as the faces CNN, with **one big difference**. The faces CNN squashed each feature map to one average (global pooling) because "is this face fake?" doesn't depend on *where* things are. The reader must say *where* the can is, so it **flattens** the 8×8 grid and keeps every position.

**Why 2 pictures, not 1.** One photo shows where the arm is, but not how fast it's moving. Two photos in a row are a tiny video: the difference between them shows the speed. That's **frame stacking**.

**Grading the eyes: regression.** The reader doesn't pick a class; it outputs numbers, so the loss is **MSE** (mean squared error). Joint angles are in radians and positions are in metres, so we **normalise** every number first (subtract the training mean, divide by the training std). That way no single number shouts louder than the rest.

**Report card v1** (validation photos): can off by 1.8 cm on average, joint speeds off by 0.53 rad/s. That sounds OK... until a brain has to *play* with those eyes.

### Ranking B: the same exam, but wearing the reader's eyes

Ranking B (`eval/ranking_b.py`) is the same exam as ranking A, with the **same real simulator** and the **same 50 sealed seeds**. Only one thing changes: the brain no longer gets the true numbers. Each step it gets the **reader's guess** from the camera picture. The score is still the *true* final distance.

So ranking B measures one thing only: **how much do imperfect eyes cost?** If B matched A, the reader would be good enough for the dream.

**What happened:** the brains fell apart. The best brain went from 5.8 cm to 20.4 cm, and every brain got 0% success. A 2 cm mistake sounds small, but it repeats 100 times, and each wrong move puts the arm somewhere the reader is even less sure about. The errors **snowball**, just like drift in a dream.

### The diagnosis: which numbers break the brain? (`eval/reader_diagnosis.py`)

Like a doctor testing one thing at a time, we let the brain see the **true** numbers except **one group**, which comes from the reader. This ran on **practice seeds**, never the exam ones, because we choose a fix from it. That kind of test is called an **ablation**.

| Group from the reader | v1: worse by |
| --- | --- |
| Joint speeds | +4.4 cm |
| Can position | +2.9 cm |
| Joint angles | +2.7 cm |
| Fingertip | +0.9 cm |
| Everything | +13.0 cm |

No single culprit: every group hurts a bit, and together they hurt far more than the sum of the parts.

### Fix attempt 1: 3 pictures (v2). It got worse.

Speeds hurt most, so v2 got 3 frames instead of 2: a longer video. Result: speeds improved a little (+3.3 cm), but angles (+6.4) and the can (+5.0) got clearly worse, and "everything" went to **+17.2 cm**. The surprise was that the can's error on stored photos was *lower* than v1's. **Lesson: a better report card on photos doesn't mean better eyes while playing.** We rejected v2.

(Ranking B's Spearman rose from 0.745 to 0.855, but every neighbouring pair was a near-tie. When everyone crawls over the finish line together, the order is luck.)

### Fix attempt 2: stop memorising (v3, next)

Both readers stopped getting better on validation photos after 4–6 epochs, while training loss kept falling. That's **overfitting**: like a student who memorises last year's paper instead of learning the subject. v3 is v1 plus **weight decay**, a gentle pull of every weight toward zero so the network can't build the huge, finely tuned weights that memorising needs. One change, as the rules say.

**Result:** v3 stopped getting *worse* on validation, but it didn't get *better*, and in closed loop it still failed (+16.2 cm). Speeds were fixed, but angles and the can broke instead. Three readers had the same report card and failed in three different ways.

### The real problem: the eyes never practised on their own mess (v4)

Imagine learning to drive only by watching a perfect driver. You never see what the road looks like when the car is drifting off it. The first time *you* drive and drift a little, you're lost, so you drift more. Our reader learned only from brains that saw the true numbers, so it never saw the strange poses its own mistakes cause. The fix is called **DAgger**: let the brains play *through the reader*, record what actually happens (with the true numbers saved), and train the reader on those pictures too.

**It worked.** We recorded 1,050 episodes of brains playing through reader v1 (`data/record_dagger.py`) and retrained the reader on old + new pictures. In the training-seed diagnosis the can's damage fell from +2.9 cm to +0.7 cm. In ranking B, **every brain landed in exactly its ranking-A place: Spearman = 1.000**, including the two brains no camera ever filmed. 

The honest small print: half of the neighbouring gaps are still near-ties, so a perfect order is partly luck, and the brains still score much worse than with true numbers (the best brain ends at 14.6 cm instead of 5.8). The eyes keep the *order* right but blur the *scores* together, and for our project the order is what counts.

**Lesson for interviews:** three tweaks to *how* the reader trained just moved the error around. Changing *what* it trained on, so it saw the situations it causes itself, fixed the ranking.

---

## Chapter 5 · The imagination (Step 5, starting)

**The job.** Show the world model the last 4 frames and the last 4 actions, and it draws **the next frame**. Do that 100 times in a row and you have a dream. This chapter is about how a network can *draw*. The reader only had to *read* numbers off a picture, which is much easier.

### Why not just predict the next frame directly?

The obvious idea is a CNN that takes past frames and spits out the next frame, trained with MSE like the reader. It works badly. When the future is a little uncertain (will the can slide 1 cm or 2 cm?), MSE rewards the **average** of the possibilities, and the average of two sharp pictures is **one blurry picture**. Feed blurry pictures back in for 100 steps and the dream melts into fog.

Diffusion models avoid this. Instead of drawing the picture in one go, they **clean it up from noise** in a few small steps. Each step only has to fix a little, so the result stays sharp.

### The core idea: a restorer of damaged photos

Imagine a photo restorer. You give them an old photo covered in grain and dust and ask, "what did the clean photo look like?"
- With **a little** grain, it's easy: just remove the specks.
- With **a lot** of grain, they must guess the big shapes first: there's an arm *roughly here*, the can *roughly there*.
- With **pure grain**, they'd have to make up the whole picture.

**Training a diffusion model = training this restorer.**
1. Take a real frame from our dataset (the true next frame).
2. Pick a random **noise level** (σ, "sigma"): sometimes tiny, sometimes huge.
3. Bury the frame in that much random noise: `noisy = frame + σ × random_noise`.
4. Ask the network to give back the **clean** frame, and grade it with MSE.

Repeat millions of times, at every noise level. The network becomes an expert restorer at every amount of damage.

**Drawing (sampling)** then works backwards. Start from **pure noise**, ask the restorer "clean this", and take a small step toward its answer. Add back a bit less noise, ask again, and repeat. Big shapes appear first, then details. Our recipe needs only **3 to 5 steps** per frame.

**Why this stays sharp:** the network never has to commit to one blurry average in one go. Each step makes a small correction, and the final steps work at fine detail, so they pick *one* sharp version instead of averaging many.

### From "a picture" to "the next picture": the hints

A plain diffusion model restores *any* plausible photo of the arm. We need *the* next frame. So the restorer also gets **hints** (this is called **conditioning**):
- **The last 4 frames**, stacked next to the noisy frame as extra colour channels. They show where everything is and how it's moving.
- **The last 4 actions** (4 × 7 = 28 numbers), squeezed by a small MLP into one vector that's added to every layer. They tell it which way the brain pushed.
- **The noise level σ**, also added to every layer, so the restorer knows how damaged its input is.

With hints like "the arm was moving left and the brain pushed left", only one clean picture fits: the true next frame. **The hints turn a picture-maker into a world model.**

### The network: a U-Net

The restorer is a **U-Net**, a CNN shaped like a U:
- **Down the left side**, the picture shrinks (64 → 32 → 16 → 8) while the channels grow. It understands *what* is where, like the reader did.
- **Up the right side**, it grows back to 64×64 to *draw* the answer.
- **Skip connections** carry fine details straight across the U from left to right, so small things like the can's edge aren't lost in the shrinking.

This is the same idea as the reader's conv blocks, plus a second half that draws.

### EDM: the recipe for the noise

**EDM** (Karras et al., 2022) is a set of careful choices about *which* noise levels to train on, how to scale the input and output so the network always sees nicely sized numbers, and how to take the sampling steps. Its practical gift is that a few steps are enough for good frames. That's why a 100-step dream is affordable on a free GPU.

### The trap we must fight later: drift (preview of Step 6)

In training, the 4 past frames are always **real** and perfect. When dreaming, they are the model's **own drawings**, slightly wrong. It's the same snowball we fought with the reader's DAgger. The fix here is **noise augmentation**: in training, deliberately add some noise to the past frames (and tell the model how much). Then it learns not to trust its context blindly. GameNGen capped this at 0.7.

### Where it lives

| Piece | Real term | Where |
| --- | --- | --- |
| The restorer | Denoiser (a U-Net, 4.5M weights) | `models/world_model.py` (`WorldModel`), U-Net copied from DIAMOND in `models/diamond_blocks.py` |
| The amount of damage | Noise level σ | `WorldModel.loss` (log-normal σ in training) |
| The hints | Conditioning: 4 frames + 4 actions + σ + context-noise level | `InnerModel.forward`: frames as channels, the rest summed into one vector |
| Practice examples | 4 frames + 4 actions -> next frame | `models/world_model_data.py` |
| Practice and grading | Training loop; error on the **moving** pixels vs "just copy the last frame" | `models/train_world_model.py` (Kaggle: `notebooks/step5_train_world_model.ipynb`) |
| Drawing from noise | Sampling, 3 Euler steps | `WorldModel.sample` |
| The noise recipe | EDM, adapted from DIAMOND | github.com/eloialonso/diamond |

**What we changed from DIAMOND (only two things).** Atari has buttons, so DIAMOND looks actions up in a table; our arm has 7 torques, so a small MLP turns the last 4 actions (28 numbers) into the hint vector. And DIAMOND never noises its past frames; we add GameNGen's noise augmentation (up to 0.7) and tell the model how much was added, to fight drift in Step 6.

**How we grade it.** Almost the whole picture is still background, so "just copy the last frame" already gets 99% of pixels right. That makes the whole-image error useless: a model that never moves the arm would look perfect. So we measure the error only on the pixels that actually change (about 1.3% of them: the arm and the can), and the model has to beat the copy-the-last-frame guess there.

**A neat EDM trick.** The network never draws the clean frame directly. It draws a *correction* to the noisy frame. When the noise is light, the noisy frame is already almost right, so the network only has to fix a little; when it's heavy, it draws almost everything. The formulas `c_skip`, `c_out` and `c_in` set that balance from σ.

**First result (3 hours on a free GPU).** After 32,000 practice rounds, the model draws the next frame so well that the error picture is almost black. On the moving pixels it is off by 12.9 brightness levels out of 255, while "just copy the last frame" is off by 30.5. Practice and exam scores are equal (train loss = val loss), so it is learning the physics, not memorising episodes. We also checked the moments that matter most: steps where the arm is pushing the can. There the model is a bit worse (16.1 instead of 11.7), but its drawings still look the same as the real frames. The catch is that the can moves only about 2 mm per step, less than one pixel, so a single step can barely show whether the model understands pushing. That's one step. Step 6 asks the hard question: what happens after 100 steps of drawing on top of its own drawings?

**The goal of Step 5:** one-step predictions on validation episodes that you can't tell from real frames by eye, *especially when the arm touches the can*. The background is learned almost instantly; contact is the hard part.

---

## Chapter 6 · The long dream (Step 6, in progress)

**The job.** Give the world model one real picture straight after a reset, then let it draw the next 100 pictures, each one from its own earlier drawings, while we replay the actions the brain really took. If the world model were perfect, the dream would match the real recording frame for frame.

**First try (8 episodes).** The good news: the pictures stay sharp all the way to frame 100. Blur, the usual failure, didn't happen. The bad news: the dream slowly wanders off. To measure *how far*, we used our state reader as a ruler. It reads the fingertip and the can off any picture, real or dreamed, and we compare that with where they really were. On real pictures the ruler is off by about 1 cm (its own error). On dream pictures the fingertip is 5 cm off by frame 10 and 38 cm by frame 100. The can follows the arm: 8 cm off at frame 25, 26 cm at frame 100.

**Why.** It's the same snowball as the reader in chapter 4. Each step is only a little wrong, but the next step is drawn on top of that little error. In training, the 4 past pictures were always perfect; in the dream they are the model's own slightly-off drawings.

**Does it obey the controls?** A world model could cheat: draw an arm that moves *plausibly* but ignores what the brain asked for. To check, we took the same start picture and gave it a *different* list of actions, both in the dream and in the real simulator. At frame 10 the dreamed arm moved 6.6 cm away from where it went before, the real arm 7.2 cm, and in the same direction in all 8 episodes. So the model does listen to the actions. Its problem is the snowball: after ~20 steps it has wandered so far that the answer no longer lines up.

**What we try next.** Prove whether noise augmentation (chapter 5's drift fix) is helping, by training the same model without it (an **ablation**). Then check whether the dream really obeys the actions, and try telling the model at dream time that its past pictures are a bit noisy.

| Piece | Real term | Where |
| --- | --- | --- |
| The dream | Autoregressive rollout | `eval/dream.py` |
| How far it wanders | Drift: PSNR vs a frozen picture, and positions read by the reader | `eval/dream.py`, `eval/dream_tracking.py` |
| Switching one piece off to prove it matters | Ablation | `configs/world_model_v1_noaug.yaml` |

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

- **Step 7 · Brains play in the dream:** dream frame, then eyes, then brain, then action, then the next dream frame. That gives **ranking C**.
- **Step 8 · The verdict:** does ranking C match ranking A? (Spearman)
- **Step 9 · Fix one thing** and measure again.
- **Step 10 · Ship:** demo, README, resume.
