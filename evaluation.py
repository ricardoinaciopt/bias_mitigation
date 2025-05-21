from pathlib import Path, PurePath
import json, re, argparse, pandas as pd
import os

GREEN, YELLOW, RESET = "\033[1;32m", "\033[33m", "\033[0m"
BOLD, UNDERLINE = "\033[1m", "\033[4m"

variant_order = [
    "none",
    "reweight_auto",
    "reweight_manual_A0.5_B1.5",
    "reweight_manual_A1.5_B0.5",
    "mitigator_demographic_parity",
    "mitigator_equalized_odds",
]

perf_high = {"roc_auc", "accuracy", "precision", "recall"}
perf_low = {"log_loss"}
fair_abs_zero = {"SPD", "EOD"}
fair_high = set()


def detect_variant(name: str) -> str:
    if "_reweight_auto" in name:
        return "reweight_auto"
    m = re.search(r"_reweight_manual_A\d+\.\d+_B\d+\.\d+", name)
    if m:
        return m.group()[1:]
    if "_mitigator_demographic_parity" in name:
        return "mitigator_demographic_parity"
    if "_mitigator_equalized_odds" in name:
        return "mitigator_equalized_odds"
    return "none"


def load_df(folder: str, setting: str | None, want_groups: bool) -> pd.DataFrame:
    pat = re.compile(
        r"metrics_(?P<setting>[^_]+)_rep(?P<rep>\d+\.\d+)_lbl(?P<lbl>\d+\.\d+)"
    )
    rows = []
    for f in Path(folder).glob("metrics_*_lbl*.json"):
        m = pat.match(PurePath(f).name)
        if not m or (setting and m["setting"] != setting):
            continue
        d = json.loads(f.read_text())
        row = {
            "setting": m["setting"],
            "rep": float(m["rep"]),
            "lbl": float(m["lbl"]),
            "variant": detect_variant(f.name),
            **d["overall_metrics"],
            **d["bias_metrics"],
        }
        if want_groups:
            for g, met in d["by_group"].items():
                for k, v in met.items():
                    row[f"{g}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def colourise(
    df: pd.DataFrame, metrics: list[str], use_color: bool, export: bool
) -> pd.DataFrame:
    out = df.copy().astype(object)
    for c in metrics:
        vals = pd.to_numeric(out[c], errors="coerce")
        if vals.dropna().empty:
            continue
        ascending = (
            c in perf_low or c in fair_abs_zero or c in {"rank_perf", "rank_fair"}
        )
        key = vals.abs() if c in fair_abs_zero else vals
        rank = key.rank(method="min", ascending=ascending)
        for i, v in vals.items():
            s = "nan" if pd.isna(v) else f"{v:.3f}"
            if use_color:
                if rank[i] == 1:
                    s = f"{GREEN}{s}{RESET}"
                elif rank[i] == 2:
                    s = f"{YELLOW}{s}{RESET}"
            elif export:
                if rank[i] == 1:
                    s = f"**{s}**"
                elif rank[i] == 2:
                    s = f"<u>{s}</u>"
            else:
                if rank[i] == 1:
                    s = f"{BOLD}{s}{RESET}"
                elif rank[i] == 2:
                    s = f"{UNDERLINE}{s}{RESET}"
            out.at[i, c] = s
    return out


def render(df: pd.DataFrame) -> str:
    try:
        from tabulate import tabulate

        return tabulate(df, headers="keys", showindex=False, tablefmt="github")
    except ImportError:
        return df.to_string(index=False)


def write_tables(
    df: pd.DataFrame,
    want_groups: bool,
    use_color: bool,
    summary_counts: dict,
    variant_order=variant_order,
    export: bool = False,
):
    core = [
        "variant",
        "roc_auc",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "SPD",
        "EOD",
        "rank_perf",
        "rank_fair",
    ]
    groups = [
        "A_accuracy",
        "B_accuracy",
        "A_recall",
        "B_recall",
        "A_precision",
        "B_precision",
        "A_approval_rate",
        "B_approval_rate",
    ]
    if want_groups:
        cols = ["variant"] + groups
    else:
        cols = core

    def metric_arrow(c):
        if c == "variant":
            return c
        if c in perf_high:
            return f"{c} (↑)"
        if c in fair_abs_zero:
            return f"{c} (~0)"
        if c in perf_low or c in {"rank_perf", "rank_fair"}:
            return f"{c} (↓)"
        if any(x in c for x in ["accuracy", "recall", "approval_rate", "precision"]):
            return f"{c} (↑)"
        return c

    header_cols = [metric_arrow(c) for c in cols]

    for (setting, rep, lbl), g in df.groupby(["setting", "rep", "lbl"]):
        # Exclude reweight_manual_A1.5_B0.5 from block if not in variant_order (i.e., if --all not passed)
        block = g.copy()
        block = block[block["variant"].isin(variant_order)]
        support_A = None
        support_B = None
        if "A_support" in block.columns and "B_support" in block.columns:
            support_A = block.iloc[0]["A_support"]
            support_B = block.iloc[0]["B_support"]
        hdr = f"\n=== {setting.upper()} | rep={rep} | lbl={lbl}"
        if support_A is not None and support_B is not None:
            hdr += f" | A_support={support_A} | B_support={support_B}"
        hdr += " ==="
        print(hdr)

        block["order"] = block["variant"].apply(variant_order.index)
        block = block.sort_values("order").drop(columns="order")

        perf_metrics = list(perf_high | perf_low)
        fair_metrics = list(fair_abs_zero | fair_high)

        sign = lambda m: 1 if (m in perf_high or m in fair_high) else -1
        block["perf_score"] = (
            block[perf_metrics].mul([sign(m) for m in perf_metrics], axis=1).sum(axis=1)
        )
        block["fair_score"] = -block[fair_metrics].abs().sum(axis=1)
        block["rank_fair"] = block["fair_score"].rank(method="min", ascending=False)

        block["rank_perf"] = block["perf_score"].rank(method="min", ascending=False)
        block["rank_fair"] = block["fair_score"].rank(method="min", ascending=False)

        # Only count for variants in variant_order (i.e., filtered)
        for v in block.loc[block["rank_perf"] == 1, "variant"]:
            if v in summary_counts:
                summary_counts[v]["best_perf"] += 1
        for v in block.loc[block["rank_perf"] == 2, "variant"]:
            if v in summary_counts:
                summary_counts[v]["second_perf"] += 1
        for v in block.loc[block["rank_fair"] == 1, "variant"]:
            if v in summary_counts:
                summary_counts[v]["best_fair"] += 1
        for v in block.loc[block["rank_fair"] == 2, "variant"]:
            if v in summary_counts:
                summary_counts[v]["second_fair"] += 1

        block = colourise(block[cols], cols[1:], use_color, export)
        block.columns = header_cols
        table = render(block)
        print(table + "\n")

    if want_groups:
        group_wins = {v: {"A": 0, "B": 0} for v in variant_order}
        for (setting, rep, lbl), g in df.groupby(["setting", "rep", "lbl"]):
            g = g[g["variant"].isin(variant_order)]  # filter for group wins as well
            for group in ["A", "B"]:
                group_metrics = [m for m in groups if m.startswith(f"{group}_")]
                for m in group_metrics:
                    vals = pd.to_numeric(g[m], errors="coerce")
                    if vals.dropna().empty:
                        continue
                    rank = vals.rank(method="min", ascending=False)
                    for i, v in g["variant"].items():
                        if pd.isna(vals[i]):
                            continue
                        if rank[i] == 1 and v in group_wins:
                            group_wins[v][group] += 1
        group_wins_df = pd.DataFrame(group_wins).T.fillna(0).astype(int)
        group_wins_df = group_wins_df.reindex(variant_order)
        group_wins_df["total"] = group_wins_df["A"] + group_wins_df["B"]
        group_wins_df = group_wins_df.sort_values(["total", "A", "B"], ascending=False)
        group_wins_df = group_wins_df.reindex(variant_order)
        group_hdr = "\n=== GROUP-BASED WINS (times ranked 1st per group) ==="
        print(group_hdr)
        print(render(group_wins_df.reset_index().rename(columns={"index": "variant"})))


def write_summary(
    summary_counts: dict,
    use_color: bool,
    variant_order=variant_order,
    export: bool = False,
    exclude_manual: bool = False,
):
    filtered_variant_order = [v for v in variant_order if v in summary_counts]
    if exclude_manual:
        filtered_variant_order = [
            v for v in filtered_variant_order if v != "reweight_manual_A1.5_B0.5"
        ]
    df_sum = pd.DataFrame(summary_counts).T.fillna(0).astype(int)
    df_sum = df_sum.reindex(filtered_variant_order)

    df_sum = df_sum.sort_values(
        ["best_perf", "best_fair", "second_perf", "second_fair"], ascending=False
    )
    df_sum = df_sum.reindex(filtered_variant_order)

    if (use_color or export) and not df_sum.empty:
        for col in ["best_perf", "second_perf", "best_fair", "second_fair"]:
            df_sum[col] = df_sum[col].astype(object)
            col_numeric = pd.to_numeric(df_sum[col], errors="coerce")
            max_val = col_numeric.max()
            second_val = col_numeric.nlargest(2).iloc[-1] if len(df_sum) > 1 else None
            for idx, row in df_sum.iterrows():
                val = str(row[col])
                if row[col] == max_val and max_val > 0:
                    if use_color:
                        df_sum.at[idx, col] = f"{GREEN}{val}{RESET}"
                    elif export:
                        df_sum.at[idx, col] = f"**{val}**"
                    else:
                        df_sum.at[idx, col] = f"{BOLD}{val}{RESET}"
                elif (
                    second_val is not None and row[col] == second_val and second_val > 0
                ):
                    if use_color:
                        df_sum.at[idx, col] = f"{YELLOW}{val}{RESET}"
                    elif export:
                        df_sum.at[idx, col] = f"<u>{val}</u>"
                    else:
                        df_sum.at[idx, col] = f"{UNDERLINE}{val}{RESET}"
        df_sum = df_sum.astype(object)

    summ_hdr = "\n=== OVERALL VARIANT SUMMARY (times ranked 1st / 2nd) ==="
    print(summ_hdr)
    print(render(df_sum.reset_index().rename(columns={"index": "variant"})))


def main():
    os.system("cls" if os.name == "nt" else "clear")
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--metrics_dir",
        default="abm_json",
        help="Base directory for metrics files (will append _static or _dynamic based on --setting)",
    )
    arg.add_argument("--eval_dir", default="evaluations")
    arg.add_argument("--setting")
    arg.add_argument("--group-eval", action="store_true")
    arg.add_argument("--color", action="store_true")
    arg.add_argument(
        "--export",
        action="store_true",
        help="Use markdown formatting for bold/underline",
    )
    arg.add_argument(
        "--vs",
        choices=[v for v in variant_order if v != "none"],
        help="Show only 'none' and the selected variant",
    )
    arg.add_argument(
        "--all",
        action="store_true",
        help="Include all variants (including reweight_manual_A1.5_B0.5)",
    )
    a = arg.parse_args()

    metrics_dir = a.metrics_dir
    if a.setting is not None:
        if a.setting == "static":
            metrics_dir = f"{metrics_dir}_static"
        else:
            metrics_dir = f"{metrics_dir}_dynamic"

    df = load_df(metrics_dir, a.setting, a.group_eval)
    if not a.all:
        df = df[df["variant"] != "reweight_manual_A1.5_B0.5"]
    if df.empty:
        print("No matching files.")
        return

    if a.vs:
        df = df[df["variant"].isin(["none", a.vs])]
        filtered_variant_order = ["none", a.vs]
    else:
        filtered_variant_order = variant_order

    summary_counts = {
        v: {"best_perf": 0, "second_perf": 0, "best_fair": 0, "second_fair": 0}
        for v in filtered_variant_order
    }

    write_tables(
        df,
        a.group_eval,
        use_color=a.color,
        summary_counts=summary_counts,
        variant_order=filtered_variant_order,
        export=a.export,
    )
    write_summary(
        summary_counts,
        use_color=a.color,
        variant_order=filtered_variant_order,
        export=a.export,
        exclude_manual=not a.all,  # pass exclusion flag
    )


if __name__ == "__main__":
    main()
