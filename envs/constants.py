"""Project-wide constants. Import these; never hard-code seeds or indices elsewhere."""

# --- Seeds -------------------------------------------------------------------
# Evaluation seeds: every policy is scored from these start positions.
# They must NEVER appear in training data, and nothing is tuned on them.
EVAL_SEEDS: list[int] = list(range(10_000, 10_050))

# Training / data-recording seeds.
TRAIN_SEEDS: list[int] = list(range(0, 10_000))

assert not set(EVAL_SEEDS) & set(TRAIN_SEEDS), "eval and train seeds overlap"

# --- Environment -------------------------------------------------------------
ENV_ID = "Pusher-v5"
EPISODE_STEPS = 100        # Pusher episodes are 100 steps
FRAME_SIZE = 64            # frames are FRAME_SIZE x FRAME_SIZE x 3, uint8
ACTION_DIM = 7             # joint torques in [-2, 2]
OBS_DIM = 23

# --- Observation layout (Pusher-v5) ------------------------------------------
JOINT_ANGLES = slice(0, 7)
JOINT_VELOCITIES = slice(7, 14)
FINGERTIP_XYZ = slice(14, 17)
OBJECT_XYZ = slice(17, 20)
GOAL_XYZ = slice(20, 23)

# --- World model / dream evaluation ------------------------------------------
CONTEXT_FRAMES = 4         # frames the world model conditions on
WARMUP_STEPS = 3           # zero-torque steps before policies take over
