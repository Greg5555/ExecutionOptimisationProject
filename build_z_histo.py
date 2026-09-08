from __future__ import division
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import os

file = r"X:\EQD\New_SGI\Scripts\CTY Project\Stats\HistoStats\df_consolide_stats.csv"

df = pd.read_csv(file)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df= df[['minute_of_day','spread_id','z_midvsSettl_roll120']]

q = [0.15,0.25,0.40,0.50,0.60,0.75,0.85]

def summarize(s):
    s = s.dropna()
    out = {"nb_obs": int(len(s))}
    for qq in q:
        out[f"z_MidvsSet_histo_q{int(qq*100):02d}"] = s.quantile(qq)

    return pd.Series(out)


z_histo_levels = (
    df.groupby(["spread_id", "minute_of_day"])
      .apply(lambda g: summarize(g["z_midvsSettl_roll120"]))
      .reset_index()
)


z_histo_levels.to_csv(r"X:\EQD\New_SGI\Scripts\CTY Project\Stats\HistoStats\Z_HistoLevels.csv", index=False)


##VISU
df_histo_zspread = pd.read_csv(r"C:\Users\gchaucho080425\Documents\Memoire\Code\HistoStats\Z_HistoLevels.csv")

OUTPUT_DIR = r"C:\Users\gchaucho080425\Documents\Memoire\Code\HistoStats\visu_z_histo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_COLS = [
    'z_MidvsSet_histo_q15',
    'z_MidvsSet_histo_q25',
    'z_MidvsSet_histo_q40',
    'z_MidvsSet_histo_q50',
    'z_MidvsSet_histo_q60',
    'z_MidvsSet_histo_q75',
    'z_MidvsSet_histo_q85',
]

for spread in df_histo_zspread['spread_id'].unique().tolist():
    df_spread = df_histo_zspread[df_histo_zspread['spread_id'] == spread].sort_values(by='minute_of_day')

    for col in Z_COLS:
        if col not in df_spread.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_spread['minute_of_day'], df_spread[col], linewidth=1.2)
        ax.set_xlabel('Minute of Day')
        ax.set_ylabel(col)
        ax.set_title(f'{col}  —  {spread}')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        safe_spread = str(spread).replace('/', '_').replace('\\', '_').replace(' ', '_')
        filename = f"{safe_spread}__{col}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close(fig)
        print(f"Saved: {filename}")

print("Done.")