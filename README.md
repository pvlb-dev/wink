# <project>

One paragraph: what this is, who it is for, and what "working" means.

The agent contract is [`AGENTS.md`](AGENTS.md). The detail lives in `AGENTS.d/`,
a separate private repo that is deliberately not part of this one.

## First five minutes in a new project

This repo is a **template**. Create the project repo from it, then:

```bash
git clone git@git.pvlb.dev:<org>/<project>.git
cd <project>
git remote set-url origin git@git.pvlb.dev:<org>/<project>.git

# AGENTS.d/ ships tracked so the clone has the auth instructions.
# Untrack it, keep the files, and let git ignore the directory from now on.
git rm -r --cached -q AGENTS.d
printf 'AGENTS.d/
' >> .gitignore

git clone git@git.pvlb.dev:agentic/<project> AGENTS.d   # when the knowledge repo exists
git branch -m master                                   # if the clone came up as main
```

Then fill the table at the top of `AGENTS.md`, write the first paragraph below,
apply the labels with `gitea_label_bootstrap {owner, repo, confirm: true}` and
open the first issue.

**Done means:** `git push` works, `git status` does not list `AGENTS.d/`, and the
first issue is labelled. Not before.
