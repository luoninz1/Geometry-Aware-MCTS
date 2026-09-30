# Geometry-Aware MCTS for Extremal Problems in Combinatorial Geometry

[Luoning Zhang](https://luoninz1.github.io/), [Xu Zhuang](https://zxmath.github.io/), [Tianhao Wang](https://tianhaow.github.io/), [Nathan Kaplan](https://www.math.uci.edu/~nckaplan/)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Framework

![Framework Overview](assets/PaperDiagram.svg)

## Installation

This project utilizes [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager written in Rust, for dependency management.

### 1. Install `uv`
If you haven't installed `uv` yet, you can do so by running:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
For Windows or alternative installation methods, refer to the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Setup the Project
Once `uv` is installed, enter the extracted source directory and sync the dependencies. `uv` will automatically create the virtual environment and install everything defined in `pyproject.toml`.

```bash
cd Geometry-Aware-MCTS
uv sync
```

## Quick Start
Run experiments from the repository root with the shared runner in `scripts/` and a configuration from `examples/`.

For example, to run the **No-Three-in-Line (Max-N3IL)** evaluation:
1. Examine or adjust `examples/example_max_n3il/config.yaml`.
2. Run the experiment using `uv`:

```bash
uv run python -m scripts.run_experiment \
  --config examples/example_max_n3il/config.yaml
```

The command must be run from the repository root. By default, generated files are written to `outputs/<example-name>/`; use `--output_dir` to override that location.

Available configurations:

- `examples/example_max_n3il/config.yaml`
- `examples/example_max_n4il/config.yaml`
- `examples/example_max_no_4_on_circle/config.yaml`
- `examples/example_max_no_isosceles/config.yaml`
- `examples/example_min_complete/config.yaml`
- `examples/example_min_geodom/config.yaml`

## Extending the Framework

Our MDP framework enables scaling and generalizability across diverse structural constraints. You can effortlessly plug your own Custom Extremal Problem into our Geometry-Aware solver.

Follow these steps to add a new environment:

### Step 1: Define the Environment Class
Create a new Python file inside `src/envs/` (e.g., `src/envs/my_custom_env.py`). Your new class should ideally inherit from `N3il_with_symmetry_and_symmetric_actions` (or another appropriate base). 

At minimum, you will need to override the following core methods to enforce the strict subset valid logic of your constraint:
- `get_valid_moves(self, state)`
- `get_valid_moves_subset(self, state, subset)`
- `get_next_state(self, state, action)`
- `simulate(self, state, num_simulations, ...)`

*Tip: For maximum performance, define the internal validity checks of these domain-specific functions using Numba (`@njit`). Refer to `src/envs/No_4_on_circle.py` for examples.*

### Step 2: Custom Terminal Conditions (Optional)
By default, the MDP framework terminates when the Feasible Action Space is an empty set (`np.sum(valid_moves) == 0`). If your specific problem variant requires a different termination condition or custom reward logic, you must also override:
- `get_value_and_terminated(self, state, valid_moves)`

*Refer to `src/envs/GeometricDominating.py` as an example of custom termination logic.*

### Step 3: Register the Environment
Make your newly defined environment visible to the framework:
1. **Import and expose** your class in `src/envs/__init__.py`.
2. **Register** the initialization hook in `src/algos/evaluate.py`. Inside the `evaluate(args)` function, add your logic to the environment conditional block:

```python
    elif args['environment'] == 'MyCustomEnv':
        n3il = MyCustomEnv(grid_size=(args['n'], args['n']), args=args, priority_grid=None)
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhang2026geometryaware,
  title   = {Geometry-Aware {MCTS} for Extremal Problems in Combinatorial Geometry},
  author  = {Zhang, Luoning and Zhuang, Xu and Wang, Tianhao and Kaplan, Nathan},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=xtlbkvz6qb}
}
```
