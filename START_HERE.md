# Start here

This folder is your project, set up so every Claude Code session starts with full context.

## One-time setup

1. Unzip this folder where you keep projects.
2. In a terminal inside it:
   ```
   git init
   git add .
   git commit -m "step 0: project scaffold"
   ```
3. Create an empty GitHub repo called `robot-world-model` and push to it.
4. Run `claude` inside the folder. Claude Code reads `CLAUDE.md` automatically every session.

## Every session

1. Open the folder and run `claude`.
2. Type `/start-session`. Claude reads your progress, tells you where you are and proposes today's plan.
3. Work through the plan one piece at a time.
4. Type `/end-session`. Claude ticks off progress, logs the session and decisions, and suggests a commit.
5. Optional: `/explain models/world_model.py` (or any file) to drill a file for interviews.

## What each file is for

| File | Purpose | Who updates it |
| --- | --- | --- |
| `CLAUDE.md` | Instructions Claude Code loads every session: goal, rules, conventions | You, rarely |
| `docs/PROJECT.md` | Problem, architecture, the three rankings | Rarely |
| `docs/BUILD_GUIDE.md` | Detailed how-to for every step | Rarely |
| `docs/PROGRESS.md` | Checklist plus session log; the "where am I" file | `/end-session` |
| `docs/DECISIONS.md` | Every design choice and why | `/end-session` |
| `.claude/skills/` | The `/start-session`, `/end-session` and `/explain` commands | Rarely |
| `envs/constants.py` | Seeds and observation indices used everywhere | Rarely |
| `configs/` | One YAML per experiment | Each experiment |

Delete this file once you're set up, or keep it for reference.
