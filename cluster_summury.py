import pandas as pd
import numpy as np

def summarize_clusters(df, cluster_col, summary_prefs, cluster_names):
    clusters = sorted(df[cluster_col].unique())
    result = {}

    for feature, method in summary_prefs.items():
        values = []
        for c in clusters:
            sub_df = df[df[cluster_col] == c]
            if df[feature].dtype == 'object' or df[feature].dtype.name == 'category':
                if method == 'count':
                    val = sub_df[feature].value_counts().to_dict()
                    val_str = "; ".join([f"{k}: {v}" for k, v in val.items()])
                    values.append(val_str)
                elif method == 'percentage':
                    val = (sub_df[feature].value_counts(normalize=True) * 100).round(2).to_dict()
                    val_str = "; ".join([f"{k}: {v}%" for k, v in val.items()])
                    values.append(val_str)
            else:
                if method == 'mean':
                    values.append(round(sub_df[feature].mean(), 2))
                elif method == 'median':
                    values.append(round(sub_df[feature].median(), 2))
                elif method == 'mode':
                    mode_val = sub_df[feature].mode()
                    values.append(mode_val[0] if not mode_val.empty else np.nan)
                elif method == 'min':
                    values.append(round(sub_df[feature].min(), 2))
                elif method == 'max':
                    values.append(round(sub_df[feature].max(), 2))
                elif method == 'sum':
                    values.append(round(sub_df[feature].sum(), 2))
        result[feature] = values

    result_df = pd.DataFrame(result, index=[cluster_names[c] for c in clusters])
    result_df = result_df.transpose()
    result_df.index = [f"{feature} ({method})" for feature, method in summary_prefs.items()]
    result_df = result_df.reset_index()
    result_df.columns = ['Feature (Method)'] + list(result_df.columns[1:])
    return result_df