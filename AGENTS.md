# AGENTS.md — wink on t14

| field | value |
|---|---|
| project | `wink` |
| purpose | Complex-valued pixel diffusion, the visual counterpart to TARDITS. Working = a checkpoint whose 28×28 samples are recognizable and a measured hyper-vs-RAT decision. |
| org / repo | `ml-lab/wink` |
| origin | `ssh://git@git-dsh/ml-lab/wink.git` (via the `git-dsh` alias, never the plain host) |
| knowledge repo | `ssh://git@git-dsh/agentic/wink.git` |
| package | `wink-diffusion` (import `wink`; v0.1.1 PyPI/Forgejo target) |

`wink` is the image sibling of `ml-lab/tardits`: same complex-valued algebra,
same config/trainer/logging conventions, new domain. It depends on
`tardits>=0.3.0` for the shared layers and the bidirectional RAT block; all
image-specific code lives here. **No VAE**: diffusion runs directly in pixel
space at 28×28 and 32×32. The first model is MNIST; CIFAR-10 is the next
milestone. M1 is done: RAT-16 (200 steps, x0, EMA 0.999, 3k CPU steps)
produces readable digits and beats matched hyper-4; the evidence is in
[`docs/findings/mnist-hyper-vs-rat.md`](docs/findings/mnist-hyper-vs-rat.md)
and [`AGENTS.d/architecture.md`](AGENTS.d/architecture.md).

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

`.forgejo/workflows/` now holds the four-release layout for this repo:

- `ci.yml` — `uv sync --locked`, compileall, imports, `pytest -q`, `uv build` +
  `twine check`; no `pull_request` trigger.
- `publish.yml` — Forgejo OIDC Authorized Integration to `ml-lab/pypi/wink-diffusion`;
  `AUDIENCE` comes from an Actions **variable**.
- `publish-pypi.yml` — public PyPI, `PYPI_TOKEN` secret, skipped when absent.
- `release.yml` — `CHANGELOG.md` section -> Forgejo release + wheel/sdist.

The only missing release action is the signed `v*` tag. **Do not tag until the
site is ready to go public.** Secrets/variables belong in repo settings, not in
a workflow or commit.

## Trigger table

Add a row the first time a task makes you read a file twice.

| if the task touches… | read first |
|---|---|
| CI, secrets, the registry, this repo's workflow set | `AGENTS.d/forge.md` |
| how it is built, the data model, the moving parts | `AGENTS.d/architecture.md` |
| how it runs, where, and how it gets updated | `AGENTS.d/deployment.md` |
| the denoiser, diffusion schedule, sampler, time rotor | `AGENTS.d/architecture.md` + `docs/model-internals.md` |
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
README.md               what this is, quickstart, honest status
pyproject.toml          hatchling; deps; wtrain/wsample console scripts
uv.lock, .python-version
.forgejo/workflows/     CI (manual dispatch + master push only)
src/wink/               namespace package: train, sample, models/, modules/
tests/                  unit + overfit smoke; no network needed
examples/configs/       mnist_rat + promptable 10k; history in AGENTS.d/examples
examples/data/          MNIST cache (11 MB), committed for a simple run
examples/saved/         empty; checkpoints are written here and gitignored
docs/                   index, model internals, repo map, experiments, findings
AGENTS.d/               private knowledge clone (ignored here, separate repo)
  GIT.md                auth and signing
  architecture.md       design decisions and port map
  forge.md              CI reality
  deployment.md         t14 CPU now, accelerator later
  gotchas.md            traps that cost a round
  input/                the drop box, gitignored contents
```

## Running it

```bash
uv sync --locked --extra mlflow
.venv/bin/pytest -q
.venv/bin/wtrain -c examples/configs/mnist_rat.yaml
.venv/bin/wsample -c examples/configs/mnist_rat.yaml \
  --checkpoint examples/saved/mnist_rat_h16/final.pt -n 64 -o samples.png
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

CPU only on this box. Set `torch_threads` explicitly in the trainer section; keep
the first MNIST runs at the matched 3,000-step budget. Kaggle/Colab runs are
blocked by the tesseract work and are not part of this milestone.

## First five minutes

The setup is already complete; this section records what was done so a fresh
clone can be checked:

```bash
git remote set-url origin ssh://git@git-dsh/ml-lab/wink.git
git config core.sshCommand "ssh -F $HOME/.ssh/config"
git config user.name dsh
git config user.email dsh@pvlb.dev
git config commit.gpgsign true
git config user.signingkey 7639CA2B2FB8F6C5D1D161F60047F55EB3D4733B
git clone ssh://git@git-dsh/agentic/wink.git AGENTS.d   # when absent
```

`AGENTS.d/` is untracked and ignored here. **Setup is complete when `git push`
works, `git status` does not list `AGENTS.d/`, and the first issue is
labelled.**

## House rule: keep this file small

The harness feeds `AGENTS.md` to the model as session instructions and silently
truncates past **65,536 bytes**. New detail goes into `AGENTS.d/`, with a trigger
row here — never inline.
