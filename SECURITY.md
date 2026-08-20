# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

- Use GitHub's private [vulnerability reporting](https://github.com/richardcsuwandi/plugbo/security/advisories/new) if it is enabled, or
- Email richardsuwandi@link.cuhk.edu.cn

Include what you observed, how to reproduce it, and the affected version or
commit. You should hear back within a week.

## What to watch for

This is a research library, not a hosted service. The usual risks are secret
leakage, not remote exploits:

- API keys in `.env`, shell history, or issue/PR text
- Campaign logs under `results/` that may include prompts or keys
- Pasting `state.json` or sandbox files that were not meant to be public

Do not commit `.env`. Do not paste keys into GitHub.
