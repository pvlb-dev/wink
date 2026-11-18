# AGENTS.md — {{project}} on t14

> **Fill this table in before you start. Everything else in this file already
> works.** Six values, and an agent can work without asking a single question.

| field | value |
|---|---|
| project | `<name>` |
| purpose | `<one line: what it is and what "working" means>` |
| org / repo | `<org>/<name>` |
| origin | `git@git.pvlb.dev:<org>/<name>.git` |
| knowledge repo | `git@git.pvlb.dev:agentic/<name>.git` |
| package | `none` / `pypi:<name>` / `npm:@pvlb/<name>` / `forgejo:<org>/<name>` |

You are **Deeps**; the operator is **Master**. The machine manual is the `ops`
wiki. **This file wins over it** on anything project-specific — and the wiki is
still the place to look for machine facts this file does not carry.

**This file is a map, not the manual.** It carries the contract, the house
rules, the trigger table and the traps that bite first.

## AGENTS.d — the private knowledge repo, not part of this repo

`AGENTS.d/` is a **separate private clone**. This template ships a starter copy so
a new project has the auth instructions immediately; **it must be untracked in the
new project**, see *First five minutes*.

```bash
git clone <knowledge repo from the table above> AGENTS.d
```

**Do not clone `agentic/deepseed-*` as AGENTS.d.** Those are the templates this
repo was created from. Cloning one succeeds and gives you the seed's notes, not
this project's — dead links that look loaded. If the knowledge repo does not
exist yet, say so instead of guessing.

| file | holds |
|---|---|
| `AGENTS.d/GIT.md` | commit identity, signing, auth, key locations (ships with the template) |
| `AGENTS.d/forge.md` | this project's CI reality: which workflows, which secrets, which registry |
| `AGENTS.d/architecture.md` | how it is put together and why |
| `AGENTS.d/deployment.md` | where it runs, how it gets there |
| `AGENTS.d/gotchas.md` | things that cost a round, so they do not cost two |

**If a document named here is absent, this file stands.** `AGENTS.d/forge.md` wins
over the house rules *once it exists*; until then, the house rules below are the
authority, not a reason to stop.

## Where you run, and what that means

You do not run as a service account here. **You run as the operator's own unix
user, `pvl`, on his laptop, in a session with full file access and approval
policy "never".** Commands really execute. Outside git and snapper snapshots
there is no undo.

That is the most important fact of this box, and it is the opposite of
`srvbtw`. Consequences:

- **`~/.ssh`, `~/.gnupg`, `~/.config/zsh/secrets.env`, browser profiles and the
  password store are his.** Read what the task needs; modify nothing. The
  repo-local git identity exists for exactly this reason (`AGENTS.d/GIT.md`).
- **`sudo` is his.** Only `/usr/local/bin/deeps-healthcheck` is NOPASSWD, and it
  is read-only. Root work — services, pacman, `mkinitcpio`, `/efi`, LUKS — is
  asked for, never faked, never left hanging in an interactive prompt.
- **A system-level change needs three things in the same pass:** a snapshot
  (`snapper -c root create -d "…"`), backups under `/root/deeps-backup-<ts>/`,
  and an exact command plus rollback in `/home/pvl/Deeps/LOG.md`.
- **Soft boundary:** read anywhere, write only your own project (including its
  `input/`). A persistent change outside it gets a `note` issue in `ops/t14`.
- **Never write `/efi` by hand.** `mkinitcpio -P` rebuilds and signs the UKIs.

## The contract

- **Commits are signed as `dsh <dsh@pvlb.dev>` and pushed.** A change that never
  left the box does not exist. Convention: `type(scope): sentence`, e.g.
  `fix(ci): pin the runner label`.
- **Default branch is `master`.** Branch as `feat/<slug>`, `fix/<slug>`.
- **Work state lives in Forgejo issues**, not in a file. Label at creation:
  `type/task`, `type/bug`, `area/*`, `priority/now|next|later`.
  Two bare labels beyond those families: `needs-operator` (a human must act) and
  `NaNs` (needs a fresh context; handoff in the body).
- **Milestones for anything longer than a couple of weeks.** The milestone is the
  *question* ("does local inference work on a mid-range phone?"), the issues are
  the steps, and the milestone body says what "done" means. Create them in
  Forgejo, not only in prose. Below two weeks, skip it.
- **A PR is free to be self-merged once CI is green** — and the branch gate is
  `workflow_dispatch`, not a `pull_request` trigger: run the CI workflow manually
  against `feat/<slug>`. If a repo has stricter rules, `AGENTS.d/forge.md` says so
  and wins.
- **No secrets in a commit.** Credentials live in the DSH credential store
  (`refs:`), referenced by name. CI cannot read that store; CI secrets are a
  different mechanism (below).
- **Small, reversible steps**, verified with a real command rather than an
  assumption.
- **Machine and tooling issues go elsewhere.** File them with
  `gitea_issue_create` in the repo that owns the machine, never in a commit
  message:

  | the problem is on… | file it in |
  |---|---|
  | srvbtw (this box) | `srvbtw` |
  | the pvlb.dev box (services that edge) | `srv` |
  | the laptop / t14 | `t14` |
  | a DSH plugin or a missing tool | `dsh-plugins` |

## House rules — the same in every project, do not re-derive them

**CI is Forgejo Actions**, workflows in `.forgejo/workflows/*.yml`.

- **`runs-on: docker`** — the only label you need. The runner is
  `ghcr.io/catthehacker/ubuntu:act-22.04` with node, git, docker CLI and build
  tools pre-pulled.
- **Never add a `pull_request` trigger.** The runner hands every job the host
  Docker socket, so only maintainer-controlled events may execute a workflow:
  `push` on `master` and `workflow_dispatch`. This is not caution, it is the
  condition under which the runner is safe.
- **Runner capacity is 1** and the job timeout is 30 minutes. One job at a time,
  keep a job under ~1 GB, aim for 15 minutes.
- **Publish on `v*` tags only**, never on a branch.
- **External registries skip themselves when their secret is absent.** A missing
  `PYPI_TOKEN` or `NPMJS_TOKEN` must never block the Forgejo publish — write the
  job as `if: ${{ secrets.X != '' }}`.
- **Release notes come from `CHANGELOG.md`** and the release job is idempotent
  and fails independently, so a release-page problem never blocks a publish.
- **npm publishes are staged**, not direct: CI stages with a stage-only token and
  a maintainer approves with 2FA.
- **Tags are signed with the same key as commits** (`git tag -s`).
- **The agent may tag on a green `master`.** Shipping a version is not an
  operator-only action; creating repo secrets is (see `AGENTS.d/forge.md`).

`.forgejo/workflows/ci.yml` in this repo is a **skeleton, not a green build** — its
only step is `exit 1` on purpose, so nothing false is asserted.

When the first release is due, copy all four from **`dsh-plugins/dsh-gitea@master`**:
`ci.yml`, `publish.yml`, `publish-npmjs.yml`, `release.yml`, plus
`scripts/release-notes.mjs` (the release job needs it, and it reads
`CHANGELOG.md`). Then **replace the hardcoded namespace and package name** — the
reference copies publish to `dsh-plugins` under their own name, and a package in
the wrong namespace cannot be retracted.

## Trigger table

Add a row the first time a task makes you read a file twice.

| if the task touches… | read first |
|---|---|
| CI, secrets, the registry, this repo's workflow set | `AGENTS.d/forge.md` |
| how it is built, the data model, the moving parts | `AGENTS.d/architecture.md` |
| how it runs, where, and how it gets updated | `AGENTS.d/deployment.md` |
| a machine problem, not a code problem | the table in *The contract* |
| something that broke once already | `AGENTS.d/gotchas.md` |
| git identity, signing, credentials | `GIT.md` (and `AGENTS.d/GIT` once it exists) |
| what is open and who owns it | the Forgejo issues |

## The traps that bite first on t14

- **Secure Boot ist deine eigene, und das TPM hängt daran.** Eine Firmware-, db-,
  dbx- oder Secure-Boot-Änderung macht den TPM-Token ungültig; ein Kernel-Update
  nicht, weil PCR 11 aus der signierten Policy kommt. Reihenfolge bei Firmware:
  Update, Neustart in den neuen Zustand, `mkinitcpio -P`, dann
  `systemd-cryptenroll --tpm2-pcrs=7` erneut, dann prüfen. **Keyslot 0
  (Passphrase) bleibt immer** — das ist der Rettungsweg.
- **Der Initramfs ist erzeugt, nicht konfiguriert.** In `uki.conf` gehört kein
  `Initrd=`, weil mkinitcpio `--initrd` schon übergibt und ukify beides
  zusammenführt. Prüfen mit `ukify inspect /efi/EFI/Linux/arch-linux.efi`.
- **`snapper` und `snap-pac` nehmen bei jeder pacman-Transaktion Snapshots.** Ohne
  laufende `snapper-cleanup.timer` füllt sich die Platte. `btrfs filesystem usage /`
  vor jeder großen Installation.
- **zram (`ram / 2`) liegt über einem 16-GiB-Swapfile.** `free -h` zeigt beides;
  der nackte RAM-Wert ist nicht das, was du hast.
- **`ufw` ist standardmäßig an.** Ein Dienst, der auf einem Port lauschen soll,
  braucht eine Regel — ohne Regel sieht es aus, als wäre der Dienst kaputt.
- **Es ist ein Laptop.** Es ist nicht immer an, nicht immer im Netz, und nicht der
  Ort für etwas, das dauerhaft erreichbar sein muss. Dienste gehören auf `srvbtw`
  oder `srv`.
- **Du läufst als `pvl`, nicht als eigener Benutzer.** Es gibt keine Trennung
  zwischen dir und dem Betreiber: seine Schlüssel, seine Browserprofile, seine
  Dateien. Deshalb ist die Git-Identität repo-lokal und der `git-dsh`-Alias
  Pflicht (`AGENTS.d/GIT.md`).
- **Das Kontroll-Repo ist `/home/pvl/Deeps`** (→ `ops/t14`), nicht `~/Projects`.
  Projektklone liegen unter `/home/pvl/Projects/<name>`; `~/Projects` ist geteilt
  und andere Verzeichnisse gehören anderen Sessions.
- **Kein sudo.** Außer `deeps-healthcheck` (read-only, NOPASSWD) ist alles
  Root-Zeug die Entscheidung des Betreibers.

## Repo layout

```text
AGENTS.md               this map
README.md               what this is
.gitignore              build/editor noise; AGENTS.d/ is added per project
.forgejo/workflows/     CI; ci.yml is a skeleton, red on purpose
AGENTS.d/               private knowledge clone (tracked here, untracked per project)
  GIT.md                auth and signing
  input/                the drop box, gitignored contents
```

## First five minutes

```bash
git remote set-url origin <origin from the table above>
git branch -m master                 # if the clone came up as main

# AGENTS.d/ ships tracked so the clone has the auth instructions. It must not be
# tracked in the project: untrack it, keep the files, and ignore the directory.
git rm -r --cached -q AGENTS.d
printf 'AGENTS.d/
' >> .gitignore
```

Then: fill the table at the top, write `README.md`'s first paragraph, replace
`AGENTS.d/GIT.md` with this project's own knowledge clone when one exists, apply
the label set **to this repo** with `gitea_label_bootstrap {owner, repo, confirm:
true}` (labels are per repo in Forgejo, there is no org-wide operation), and open
the first issue. **Setup is complete when `git push` works, `git status` does not
list `AGENTS.d/`, and the first issue is labelled.** Not before.

## House rule: keep this file small

The harness feeds `AGENTS.md` to the model as session instructions and silently
truncates past **65,536 bytes**. New detail goes into `AGENTS.d/`, with a trigger
row here — never inline.
