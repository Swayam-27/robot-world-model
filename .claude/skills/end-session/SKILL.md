---
name: end-session
description: Close a work session on the robot arm world model: update progress and decisions, and prepare a commit. Use when the user types /end-session or says they are done for today.
---

Close the session:

1. Tick the finished tasks in `docs/PROGRESS.md`. If a whole step is done, update the **Current step** line at the top.
2. Add a session-log entry at the top of the log in `docs/PROGRESS.md`:
   `### YYYY-MM-DD` then 3 bullets: Done, Next, Blockers. Include any real numbers we measured (never estimates).
3. If we made or changed a design choice or experiment setting, append it to `docs/DECISIONS.md` as `YYYY-MM-DD · decision · why · alternatives`.
4. Show `git status`, then propose a commit message in the form `step N: <what changed>`. Commit only after I confirm.
5. End with 3 quick questions about today's code that an interviewer might ask, without the answers, so I can test myself.
