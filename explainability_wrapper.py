import os
import shap
import shapiq
from shapiq.plot import stacked_bar_plot, upset_plot, network_plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

SHAP_SUBSET = 250
SECOND_ORDER_INSTANCES = 20


def explainability_analysis(
    model,
    X_train,
    X_test,
    output_dir=None,
    prefix="model",
    save_interaction_values=False,
):
    os.makedirs(output_dir, exist_ok=True)
    feature_names = list(X_test.columns)
    assert list(X_train.columns) == feature_names

    # if the test set is too large, we will only use the last SHAP_SUBSET instances (only applicable in first-order SHAP)
    # if X_test.shape[0] > SHAP_SUBSET:
    #     X_test = X_test.iloc[-SHAP_SUBSET:]
    #     logging.info(
    #         "Using only the last %d instances of the test set for analysis. New size: %d",
    #         SHAP_SUBSET,
    #         X_test.shape[0],
    #     )

    if X_train.shape[0] > X_test.shape[0]:
        common_rows = X_train.merge(X_test, how="inner")
        if not common_rows.empty:
            logging.info(
                "Found %d common rows between training and test sets. Removing them from training set.",
                common_rows.shape[0],
            )
            X_train = X_train[~X_train.isin(common_rows).all(axis=1)]

        def _predict_helper(X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=feature_names)
            pred = model.predict(X)
            if isinstance(pred, pd.DataFrame):
                pred = pred.values
            # if pred is a (n_samples, 2) array, take the second column / positive class
            if isinstance(pred, np.ndarray) and pred.ndim == 2 and pred.shape[1] == 2:
                pred = pred[:, 1]
            return pred

    # Standard SHAP (not used in the paper, but kept for reference)
    # logging.info("Computing first-order SHAP values")
    # try:
    #     expl = shap.Explainer(_predict_helper, X_train)
    #     sv = expl(X_test)
    #     plt.figure()
    #     shap.summary_plot(
    #         sv, X_test, feature_names=feature_names, max_display=5, show=False
    #     )
    #     shp_path = os.path.join(output_dir, f"{prefix}_shap_summary.pdf")
    #     plt.savefig(shp_path, bbox_inches="tight", dpi=500)
    #     plt.close()
    #     logging.info("SHAP summary saved to %s", shp_path)
    # except Exception as e:
    #     logging.error("SHAP failed: %s", e)

    # SHAP-IQ second-order
    logging.info("Computing second-order interactions with SHAP-IQ")
    try:
        si_expl = shapiq.Explainer(
            _predict_helper,
            data=X_train.to_numpy(),
            index="SII",
            max_order=2,
            random_state=42,
        )
        # shap_iq allows only explanations of interactions (second order) for one instance at a time. Using only the first SECOND_ORDER_INSTANCES instances of X_test.
        for idx, row in X_test.head(SECOND_ORDER_INSTANCES).iterrows():
            element = "instance"
            try:
                logging.info("Explaining %s %d", element, idx)

                iv = si_expl.explain(row.to_numpy())
                if save_interaction_values:
                    np.save(
                        os.path.join(
                            output_dir, f"{prefix}_interaction_values_{idx}.npy"
                        ),
                        iv,
                    )
                # stacked‐bar for this instance
                # try:
                #     logging.info("Generating stacked-bar plot for %s %d", element, idx)
                #     out_bar = stacked_bar_plot(
                #         iv, feature_names=feature_names, show=False
                #     )
                #     fig_bar = out_bar[0] if isinstance(out_bar, tuple) else out_bar
                #     fig_bar.savefig(
                #         os.path.join(
                #             output_dir, f"{prefix}_{element}{idx}_stacked_bar.pdf"
                #         ),
                #         bbox_inches="tight",
                #         dpi=500,
                #     )
                #     plt.close(fig_bar)
                # except Exception as e:
                #     logging.error(
                #         "Failed to generate stacked-bar plot for instance %d: %s",
                #         idx,
                #         e,
                #     )

                # upset plot
                # try:
                #     logging.info("Generating upset plot for %s %d", element, idx)
                #     out_up = upset_plot(
                #         iv,
                #         feature_names=feature_names,
                #         color_matrix=True,
                #         show=False,
                #     )
                #     fig_up = out_up[0] if isinstance(out_up, tuple) else out_up
                #     fig_up.savefig(
                #         os.path.join(
                #             output_dir, f"{prefix}_{element}{idx}_upset_plot.pdf"
                #         ),
                #         bbox_inches="tight",
                #         dpi=500,
                #     )
                #     plt.close(fig_up)
                # except Exception as e:
                #     logging.error(
                #         "Failed to generate upset plot for instance %d: %s", idx, e
                #     )

                # network plot
                try:
                    logging.info("Generating network plot for %s %d", element, idx)

                    with plt.rc_context({"font.size": 18}):
                        out_net = network_plot(
                            iv,
                            feature_names=feature_names,
                            draw_legend=False,
                            show=False,
                        )
                    fig_net = out_net[0] if isinstance(out_net, tuple) else out_net
                    fig_net.savefig(
                        os.path.join(
                            output_dir, f"{prefix}_{element}{idx}_network_plot.png"
                        ),
                        bbox_inches="tight",
                        dpi=1000,
                    )
                    plt.close(fig_net)
                except Exception as e:
                    logging.error(
                        "Failed to generate network plot for instance %d: %s", idx, e
                    )
            except Exception as e:
                logging.error("Failed to explain instance %d: %s", idx, e)
        logging.info("SHAP-IQ plots saved to %s", output_dir)
    except Exception as e:
        logging.error("SHAP-IQ failed: %s", e)

    logging.info("Explainability complete.")
