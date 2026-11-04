---
name: start-session
description: Start a work session on the robot arm world model. Use at the beginning of every session, or when the user types /start-session.
---

Start a work session:

1. Read `docs/PROGRESS.md`. Find the current step and the first unticked task. Read the last 2 session-log entries.
2. Read only that step's section in `docs/BUILD_GUIDE.md`.
3. Skim `docs/DECISIONS.md` for anything that affects this step.
4. Run `git status` and `git log --oneline -5` to see where the code stands.
5. Reply briefly with:
   - where we are (step, last session's result, any open blocker)
   - today's plan: 2 to 4 concrete tasks that fit about 2 hours
   - the one concept I need to understand for today's work, in 2 sentences
6. Wait for my go-ahead before writing code.
