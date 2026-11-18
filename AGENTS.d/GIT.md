# GIT.md — the agent's git identity on t14

Part of the manual — the map is [`AGENTS.md`](../AGENTS.md). Source of truth for
the setup: `ops/t14` wiki, page **DSH Bootcamp**.

## The difference from the server

On `srvbtw` the `dsh` identity is **global**. Here it is **repo-local**, because
the global git identity belongs to the operator. Set it per repository, never
globally, and **never author a commit as `Pavel <pvlb.dev@gmail.com>`** — those
are his commits, not yours to impersonate or rewrite.

## Bootstrap a checkout

```bash
git config user.name  dsh
git config user.email dsh@pvlb.dev
git config commit.gpgsign true
git config user.signingkey 7639CA2B2FB8F6C5D1D161F60047F55EB3D4733B
git remote add origin ssh://git@git-dsh/<owner>/<repo>.git

git log -1 --format='%an <%ae> sig=%G?'   # ... sig=G
ssh -T git@git-dsh                        # "Hi there, dsh!"
```

| | |
|---|---|
| Forgejo account | `dsh` — org member, non-admin (`pvl` is the admin) |
| Commit identity | `dsh <dsh@pvlb.dev>`, **per repository**, never global |
| Signing | GPG, key `7639CA2B2FB8F6C5D1D161F60047F55EB3D4733B`, commits show *Verified* |
| SSH key | `~/.ssh/dsh_ed25519`, registered on the `dsh` account |
| SSH alias | `~/.ssh/dsh.config` maps **`git-dsh`** → `git.pvlb.dev`, `IdentitiesOnly yes` |
| Remote | `ssh://git@git-dsh/<owner>/<repo>.git` |

## Why the alias is not optional

Plain `git.pvlb.dev` resolves to the operator's Bitwarden-backed keys. The
`git-dsh` alias pins yours. **Never push through the plain hostname**, never copy
his keys into your config, never add an `IdentityFile` for the bare host.

## Credentials

| Where | What |
|---|---|
| `~/.dsh/.credentials.yaml` (0600) | DSH credential store: `GITEA_TOKEN`, `DEEPSEEK_API_KEY` |
| `~/.config/zsh/secrets.env` (0600) | shell tokens, sourced by zshrc |

- The `gitea_*` tools resolve `GITEA_TOKEN` **at call time**; rotating means
  editing that file.
- **Never** put a token in `.bashrc`, `.zshrc`, `env`, a committed `.env`, or a
  chat message.

## Your own key, in the only keyring there is

You have your **own** GPG key for `dsh <dsh@pvlb.dev>` (`7639CA2B…`). Because t14
has a single unix user, that key sits in the same `~/.gnupg` as the operator's —
same directory, different key. It is your key, not a borrowed one.

- **Do not touch the directory.** No key generation, no imports, no changes to
  gpg options or `gpg.program`.
- **Your key has no passphrase. A passphrase prompt is a configuration bug, not a
  step.** It means `user.signingkey` is unset and gpg is offering the operator's
  keys first. Fix it with the `git config` line above; do **not** learn to cancel
  the prompt and let it time out — that works only while someone is watching, and
  fails silently in an unattended run.
- If the `dsh` key is missing or refuses, **say so**. Never generate a
  replacement: the commits would be signed by a key the forge does not know, and
  every one of them would show unverified.

## History

**Never rewrite published history** — no force-push to `master`. A commit that
never left the box does not exist for Master.
