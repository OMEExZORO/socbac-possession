@AGENTS.md

## Claude Code specifics

`.claude/settings.json` denies `Read`/`Edit` on `data/**` and on match-data file patterns anywhere.
It also denies shell commands that mention those paths. These rules are a backstop. The real rule
is section 1 of AGENTS.md: they cannot catch a script that prints data, so never write one.
