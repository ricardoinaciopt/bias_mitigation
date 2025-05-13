import subprocess
import json
import itertools
import argparse

parser = argparse.ArgumentParser(description="Run bias mitigation pipeline.")
parser.add_argument(
    "--xai", action="store_true", help="Enable XAI flag in configurations."
)
args = parser.parse_args()

lbl_list = [0.0, 0.4, 0.5, 0.6]
rep_list = [0.5, 0.6, 0.7, 0.8]
group_weights_list = [
    None,
    "auto",
    [{"A": 0.5, "B": 1.5}, {"A": 1.5, "B": 0.5}],
]
fairness_constraints = ["demographic_parity", "equalized_odds"]

configs = [
    {
        "setting": setting,
        "lbl": lbl,
        "rep": rep,
        "group_weights": group_weights,
        "use_mitigator": use_mitigator,
        "fairness_constraint": fairness_constraint,
        "xai": args.xai,  # Use parsed xai flag
    }
    for setting, lbl, rep, group_weights, use_mitigator, fairness_constraint in itertools.product(
        # ["static", "dynamic"],
        ["static"],
        lbl_list,
        rep_list,
        group_weights_list,
        [False, True],
        fairness_constraints,
    )
    if not (
        use_mitigator and group_weights is not None
    )  # no reweighting with mitigation
]


def run_pipeline(
    setting, lbl, rep, group_weights, use_mitigator, fairness_constraint, xai
):
    base_cmd = [
        "python",
        "main_abm_pipeline.py",
        "--setting",
        setting,
        "--lbl",
        str(lbl),
        "--rep",
        str(rep),
    ]

    if xai:
        base_cmd.append("--xai")
    if use_mitigator:
        base_cmd += [
            "--mitigator",
            "--fairness-constraint",
            fairness_constraint,
        ]
        print(f"Running: {' '.join(base_cmd)}")
        subprocess.run(base_cmd, check=True)

    elif group_weights == "auto":
        cmd = base_cmd + ["--reweighting"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    else:
        if group_weights is not None:
            for weights in group_weights:
                weights_str = json.dumps(weights)
                cmd = base_cmd + ["--reweighting", "--group-weights", weights_str]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
        else:
            # No group weights specified, just run base command
            print(f"Running: {' '.join(base_cmd)}")
            subprocess.run(base_cmd, check=True)


def main():
    for cfg in configs:
        run_pipeline(
            cfg["setting"],
            cfg["lbl"],
            cfg["rep"],
            cfg["group_weights"],
            cfg["use_mitigator"],
            cfg["fairness_constraint"],
            cfg["xai"],  # Pass xai flag to run_pipeline
        )


if __name__ == "__main__":
    main()
