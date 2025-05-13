from pathlib import Path, PurePath
import json, re, argparse, pandas as pd

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


def load_df(folder: str, model: str | None, want_groups: bool) -> pd.DataFrame:
    pat = re.compile(
        r"metrics_(?P<model>[^_]+)_rep(?P<rep>\d+\.\d+)_lbl(?P<lbl>\d+\.\d+)"
    )
    rows = []
    for f in Path(folder).glob("metrics_*_lbl*.json"):
        m = pat.match(PurePath(f).name)
        if not m or (model and m["model"] != model):
            continue
        d = json.loads(f.read_text())
        row = {
            "model": m["model"],
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


def colourise(df: pd.DataFrame, metrics: list[str], use_color: bool) -> pd.DataFrame:
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
    path_txt: Path,
    use_color: bool,
    summary_counts: dict,
    variant_order=variant_order,
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
        # For group-wise metrics, use ↑ for accuracy/recall/approval_rate/precision
        if any(x in c for x in ["accuracy", "recall", "approval_rate", "precision"]):
            return f"{c} (↑)"
        return c

    header_cols = [metric_arrow(c) for c in cols]

    with open(path_txt, "w", encoding="utf-8") as fh:
        for (model, rep, lbl), g in df.groupby(["model", "rep", "lbl"]):
            support_A = None
            support_B = None
            if "A_support" in g.columns and "B_support" in g.columns:
                support_A = g.iloc[0]["A_support"]
                support_B = g.iloc[0]["B_support"]
            hdr = f"\n=== {model.upper()} | rep={rep} | lbl={lbl}"
            if support_A is not None and support_B is not None:
                hdr += f" | A_support={support_A} | B_support={support_B}"
            hdr += " ==="
            print(hdr)
            fh.write(hdr + "\n")

            block = g.copy()
            block["order"] = block["variant"].apply(variant_order.index)
            block = block.sort_values("order").drop(columns="order")

            perf_metrics = list(perf_high | perf_low)
            fair_metrics = list(fair_abs_zero | fair_high)

            sign = lambda m: 1 if (m in perf_high or m in fair_high) else -1
            block["perf_score"] = (
                block[perf_metrics]
                .mul([sign(m) for m in perf_metrics], axis=1)
                .sum(axis=1)
            )
            block["fair_score"] = -block[fair_metrics].abs().sum(axis=1)
            block["rank_fair"] = block["fair_score"].rank(method="min", ascending=False)

            block["rank_perf"] = block["perf_score"].rank(method="min", ascending=False)
            block["rank_fair"] = block["fair_score"].rank(method="min", ascending=False)

            for v in block.loc[block["rank_perf"] == 1, "variant"]:
                summary_counts[v]["best_perf"] += 1
            for v in block.loc[block["rank_perf"] == 2, "variant"]:
                summary_counts[v]["second_perf"] += 1
            for v in block.loc[block["rank_fair"] == 1, "variant"]:
                summary_counts[v]["best_fair"] += 1
            for v in block.loc[block["rank_fair"] == 2, "variant"]:
                summary_counts[v]["second_fair"] += 1

            block = colourise(block[cols], cols[1:], use_color)
            block.columns = header_cols
            table = render(block)
            print(table + "\n")
            fh.write(table + "\n")

        if want_groups:
            group_wins = {v: {"A": 0, "B": 0} for v in variant_order}
            for (model, rep, lbl), g in df.groupby(["model", "rep", "lbl"]):
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
                            if rank[i] == 1:
                                group_wins[v][group] += 1
            # Prepare summary DataFrame
            group_wins_df = pd.DataFrame(group_wins).T.fillna(0).astype(int)
            group_wins_df = group_wins_df.reindex(variant_order)
            group_wins_df["total"] = group_wins_df["A"] + group_wins_df["B"]
            group_wins_df = group_wins_df.sort_values(
                ["total", "A", "B"], ascending=False
            )
            group_hdr = "\n=== GROUP-BASED WINS (times ranked 1st per group) ==="
            print(group_hdr)
            print(
                render(group_wins_df.reset_index().rename(columns={"index": "variant"}))
            )
            fh.write(group_hdr + "\n")
            fh.write(
                render(group_wins_df.reset_index().rename(columns={"index": "variant"}))
                + "\n"
            )


def write_summary(
    summary_counts: dict, eval_path: Path, use_color: bool, variant_order=variant_order
):
    df_sum = pd.DataFrame(summary_counts).T.fillna(0).astype(int)
    df_sum = df_sum.reindex(variant_order)

    df_sum["total_best"] = df_sum["best_perf"] + df_sum["best_fair"]
    df_sum["total_second"] = df_sum["second_perf"] + df_sum["second_fair"]
    df_sum = df_sum.sort_values(["total_best", "total_second"], ascending=False)

    if use_color and not df_sum.empty:
        best_tot = pd.to_numeric(df_sum["total_best"], errors="coerce").max()
        second_tot = (
            pd.to_numeric(df_sum["total_best"], errors="coerce").nlargest(2).iloc[-1]
            if len(df_sum) > 1
            else None
        )

        for col in ["best_perf", "second_perf", "best_fair", "second_fair"]:
            df_sum[col] = df_sum[col].astype(object)
            col_numeric = pd.to_numeric(df_sum[col], errors="coerce")
            max_val = col_numeric.max()
            second_val = col_numeric.nlargest(2).iloc[-1] if len(df_sum) > 1 else None
            for idx, row in df_sum.iterrows():
                val = str(row[col])
                if row[col] == max_val and max_val > 0:
                    df_sum.at[idx, col] = f"{GREEN}{val}{RESET}"
                elif (
                    second_val is not None and row[col] == second_val and second_val > 0
                ):
                    df_sum.at[idx, col] = f"{YELLOW}{val}{RESET}"
        df_sum = df_sum.astype(object)
        total_best_numeric = pd.to_numeric(df_sum["total_best"], errors="coerce")
        total_second_numeric = pd.to_numeric(df_sum["total_second"], errors="coerce")
        best_tot = total_best_numeric.max()
        second_tot = (
            total_best_numeric.nlargest(2).iloc[-1] if len(df_sum) > 1 else None
        )
        best_tot_second = total_second_numeric.max()
        second_tot_second = (
            total_second_numeric.nlargest(2).iloc[-1] if len(df_sum) > 1 else None
        )
        for idx, row in df_sum.iterrows():
            val_best = str(row["total_best"])
            if pd.to_numeric(row["total_best"], errors="coerce") == best_tot:
                df_sum.at[idx, "total_best"] = f"{GREEN}{val_best}{RESET}"
            elif (
                second_tot is not None
                and pd.to_numeric(row["total_best"], errors="coerce") == second_tot
            ):
                df_sum.at[idx, "total_best"] = f"{YELLOW}{val_best}{RESET}"
            # Highlight total_second
            val_second = str(row["total_second"])
            if pd.to_numeric(row["total_second"], errors="coerce") == best_tot_second:
                df_sum.at[idx, "total_second"] = f"{GREEN}{val_second}{RESET}"
            elif (
                second_tot_second is not None
                and pd.to_numeric(row["total_second"], errors="coerce")
                == second_tot_second
            ):
                df_sum.at[idx, "total_second"] = f"{YELLOW}{val_second}{RESET}"

    summ_hdr = "\n=== OVERALL VARIANT SUMMARY (times ranked 1st / 2nd) ==="
    print(summ_hdr)
    print(render(df_sum.reset_index().rename(columns={"index": "variant"})))
    with open(eval_path / "evaluation_report.txt", "a", encoding="utf-8") as f:
        f.write(summ_hdr + "\n")
        f.write(
            render(df_sum.reset_index().rename(columns={"index": "variant"})) + "\n"
        )


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--metrics_dir", default="abm_json")
    arg.add_argument("--eval_dir", default="evaluations")
    arg.add_argument("--model")
    arg.add_argument("--group-eval", action="store_true")
    arg.add_argument("--color", action="store_true")
    arg.add_argument(
        "--vs",
        choices=[v for v in variant_order if v != "none"],
        help="Show only 'none' and the selected variant",
    )
    a = arg.parse_args()

    df = load_df(a.metrics_dir, a.model, a.group_eval)
    if df.empty:
        print("No matching files.")
        return

    if a.vs:
        df = df[df["variant"].isin(["none", a.vs])]
        filtered_variant_order = ["none", a.vs]
    else:
        filtered_variant_order = variant_order

    eval_path = Path(a.eval_dir)
    eval_path.mkdir(exist_ok=True)
    df.to_csv(eval_path / "evaluation_summary.csv", index=False)

    summary_counts = {
        v: {"best_perf": 0, "second_perf": 0, "best_fair": 0, "second_fair": 0}
        for v in filtered_variant_order
    }

    write_tables(
        df,
        a.group_eval,
        eval_path / "evaluation_report.txt",
        use_color=a.color,
        summary_counts=summary_counts,
        variant_order=filtered_variant_order,
    )
    write_summary(
        summary_counts,
        eval_path,
        use_color=a.color,
        variant_order=filtered_variant_order,
    )


if __name__ == "__main__":
    main()
