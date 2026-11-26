# Repository map

```text
wink/
├── AGENTS.md                  # agent map: contract, triggers, traps
├── README.md                  # what WINK is, quickstart, status
├── LICENSE                    # Apache-2.0
├── pyproject.toml             # hatchling; deps; wtrain/wsample; CPU torch index
├── uv.lock, .python-version   # Python 3.13, locked environment
├── .forgejo/workflows/ci.yml  # uv sync, compileall, imports, pytest
│
├── src/wink/
│   ├── train.py               # wtrain CLI
│   ├── sample.py              # wsample CLI
│   ├── models/
│   │   ├── base.py            # ImageModelConfig + ImageTransformerBase
│   │   ├── registry.py        # hyper | rat, aliases, checkpoint loader
│   │   ├── hyper.py           # non-causal complex softmax attention
│   │   └── rat.py             # tardits RATBlock adapter
│   └── modules/
│       ├── patchify.py        # Patchify/Unpatchify + 2D position
│       ├── diffusion.py       # schedule, losses, DDIM/DDPM, time rotor
│       ├── data.py            # MNIST/CIFAR-10 loaders, synthetic data
│       ├── trainer.py         # training loop, gamma projection, checkpoints
│       └── logging.py         # console + MLflow, sample grids
│
├── tests/                     # 38 unit/integration tests, no network needed
├── benchmarks/bench_step.py   # ms/step and parameter count
├── examples/
│   ├── configs/               # mnist_rat + promptable 10k (two entry points)
│   ├── data/                  # MNIST cache (11 MB), committed
│   └── saved/                 # empty; checkpoints/grids are gitignored
├── docs/
│   ├── index.md, model-internals.md, repo-map.md
│   ├── experiments/           # dated run reports
│   ├── findings/              # decision notes
│   └── assets/                # committed sample grids/curves
└── AGENTS.d/                  # private knowledge clone, ignored here
```

## Commands

```bash
# environment
uv sync --locked --extra mlflow

# tests
.venv/bin/pytest -q

# benchmark one training step (no dataset)
.venv/bin/python benchmarks/bench_step.py -c examples/configs/mnist_rat.yaml

# train
.venv/bin/wtrain -c examples/configs/mnist_rat.yaml
.venv/bin/wtrain -c examples/configs/mnist_rat_cond_cfg_h32_10k.yaml

# sample
.venv/bin/wsample -c examples/configs/mnist_rat.yaml \
  --checkpoint examples/saved/mnist_rat_h16/final.pt \
  -n 64 --steps 50 -o samples.png

# inspect MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Config sections

Each YAML section maps to a dataclass next to its owner:

| section | dataclass | file |
|---|---|---|
| `model` | `ImageModelConfig` | `src/wink/models/base.py` |
| `diffusion` | `DiffusionConfig` | `src/wink/modules/diffusion.py` |
| `trainer` | `TrainerConfig` | `src/wink/modules/trainer.py` |
| `mlflow` | `MLFlowConfig` | `src/wink/modules/logging.py` |
| `data` | `DataConfig` | `src/wink/modules/data.py` |

`model.arch` is popped before `ImageModelConfig(**...)`; unknown keys fail loudly.
