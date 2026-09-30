import argparse
from collections import Counter
from pathlib import Path

import yaml

from src.algos import MCTS, evaluate
from src.rewards.n3il_rewards import set_reward_strategy
from src.utils.clear_pycache import clear_pycache


REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate_num_searches(num_searches_str, n):
    """Evaluate a search-count expression that may depend on ``n``."""
    try:
        safe_env = {"n": n}
        result = eval(str(num_searches_str), {"__builtins__": None}, safe_env)
        return int(result)
    except Exception as exc:
        try:
            return int(num_searches_str)
        except (TypeError, ValueError):
            raise ValueError(
                f"Failed to evaluate num_searches expression "
                f"'{num_searches_str}': {exc}"
            ) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MCTS experiments with a YAML config and CLI overrides."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file, relative to the repository root.",
    )
    parser.add_argument("--start", type=int, default=None, help="Starting value of n (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Ending value of n (exclusive)")
    parser.add_argument("--step", type=int, default=None, help="Step size for n values")
    parser.add_argument("--repeat", type=int, default=None, help="Number of runs for each n value")
    parser.add_argument("--symmetric_action", type=str, default=None, help="Symmetric action mode")
    parser.add_argument("--environment", type=str, default=None, help="Environment name")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--reward",
        type=str,
        default=None,
        choices=["default", "exp_reverse", "exp_growth", "linear", "gaussian", "optimal_3x3"],
        help="Reward function strategy to use",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument(
        "--num_searches",
        type=str,
        default=None,
        help="Number of searches; may be an expression of n",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    return parser


def resolve_from_repo(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(config_path, parser):
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        parser.error(f"Invalid YAML in configuration file {config_path}: {exc}")

    if not isinstance(config, dict):
        parser.error(f"Configuration must be a YAML mapping: {config_path}")
    return config


def resolve_output_dir(config, config_path):
    configured_output = config.get("output_dir")
    if configured_output:
        return resolve_from_repo(configured_output)
    return REPO_ROOT / "outputs" / config_path.parent.name


def run_experiment(config, output_dir):
    print(f"Configuration: Setting Reward Strategy to '{config.get('reward', 'exp_growth')}'")
    set_reward_strategy(config.get("reward", "exp_growth"))

    table_dir = output_dir
    figure_dir = output_dir / "figure"
    web_viz_dir = output_dir / "web_visualization"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if config.get("symmetric_action") == "None":
        config["symmetric_action"] = None

    n_list = range(config["start"], config["end"], config["step"])
    MCTS.clear_global_data()
    all_results = {n: [] for n in n_list}

    for trial_index in range(config["repeat"]):
        print(f"\n=== Starting Trial {trial_index + 1}/{config['repeat']} ===")

        for n in n_list:
            print(f"Trial {trial_index + 1}/{config['repeat']} for n={n}...", end=" ")
            symmetric_action = config.get("symmetric_action")
            args = {
                "environment": config.get("environment"),
                "algorithm": config.get("algorithm"),
                "save_optimal_terminal_state": config.get("save_optimal_terminal_state", True),
                "save_all_optimal_terminal_states": n < 11,
                "symmetric_action": symmetric_action,
                "node_compression": config.get("node_compression", True),
                "max_level_to_use_symmetry": 2 * n if symmetric_action else 1,
                "n": n,
                "C": config.get("C", 1.41),
                "num_searches": evaluate_num_searches(config.get("num_searches"), n),
                "num_workers": config.get("num_workers", 1),
                "virtual_loss": config.get("virtual_loss", 1.0),
                "process_bar": config.get("process_bar", False),
                "display_state": config.get("display_state", True),
                "logging_mode": config.get("logging_mode", True),
                "light_log": config.get("light_log", True),
                "TopN": n,
                "simulate_with_priority": config.get("simulate_with_priority", False),
                "table_dir": str(table_dir),
                "figure_dir": str(figure_dir),
                "random_seed": (
                    config.get("random_seed")
                    if config.get("random_seed") is not None
                    else trial_index
                ),
                "tree_visualization": config.get("tree_visualization", False),
                "pause_at_each_step": config.get("pause_at_each_step", False),
                "continue_from_existing_state": config.get("continue_from_existing_state"),
            }

            num_points = evaluate(args)
            all_results[n].append(num_points)
            print(f"Final points: {num_points}")

    for n in n_list:
        results_for_n = all_results[n]
        print(f"\n--- Summary for n={n} ---")
        print(f"Results: {results_for_n}")
        if results_for_n:
            print(f"Min points: {min(results_for_n)}")
            print(f"Max points: {max(results_for_n)}")
            print(f"Average points: {sum(results_for_n) / len(results_for_n):.2f}")
            print(f"Median points: {sorted(results_for_n)[len(results_for_n) // 2]}")
        else:
            print("No results collected.")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF ALL EXPERIMENTS")
    print("=" * 60)
    for n, results in all_results.items():
        print(f"Grid size n={n}: {results}")
        if results:
            print(
                f"  → Min: {min(results)}, Max: {max(results)}, "
                f"Avg: {sum(results) / len(results):.4f}"
            )
            frequencies = Counter(results)
            print(f"  → Frequency: {dict(frequencies)}")
            if n == 3:
                optimal_count = frequencies.get(4, 0)
                print(
                    f"  → Found optimal 4-point solution: {optimal_count}/{len(results)} "
                    f"times ({100 * optimal_count / len(results):.1f}%)"
                )
        print()

    if config.get("tree_visualization", False):
        print("Note: Tree visualizations were generated for each trial and step.")
        experiment_name = (
            f"mcts_n{config.get('start')}-{config.get('end') - 1}"
            f"_trials{config.get('repeat')}"
        )
        final_html = MCTS.save_final_visualization(str(web_viz_dir), experiment_name)
        if final_html:
            print(f"Comprehensive visualization saved to: {final_html}")
        else:
            print("No visualization data found.")


def main(argv=None):
    parser = build_parser()
    cli_args = parser.parse_args(argv)
    config_path = resolve_from_repo(cli_args.config)
    config = load_config(config_path, parser)

    for key, value in vars(cli_args).items():
        if value is not None and key != "config":
            config[key] = value

    output_dir = resolve_output_dir(config, config_path)
    clear_pycache(str(REPO_ROOT / "src"))
    run_experiment(config, output_dir)


if __name__ == "__main__":
    main()
