import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
from importlib import import_module

gr = import_module("gradio")

def draw_court(ax, color="black", lw=1.2):
    ax.add_patch(Circle((0, 0), 7.5, lw=lw, ec=color, fill=False))
    ax.add_patch(Rectangle((-30, -7.5), 60, -1, lw=lw, ec=color, fc=color))
    ax.add_patch(Rectangle((-80, -47.5), 160, 190, lw=lw, ec=color, fill=False))
    ax.add_patch(Rectangle((-60, -47.5), 120, 190, lw=lw, ec=color, fill=False))
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, lw=lw, ec=color))
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=180, theta2=360, lw=lw, ec=color, ls="dashed"))
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, lw=lw, ec=color))
    ax.plot([-220, -220], [-47.5, 92.5], color=color, lw=lw)
    ax.plot([220, 220], [-47.5, 92.5], color=color, lw=lw)
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, lw=lw, ec=color))
    ax.set_xlim(-250, 250); ax.set_ylim(-50, 300); ax.set_aspect("equal"); ax.axis("off")

DATA = pd.read_csv("shots_meta_preds.csv")
DATA = DATA[DATA["is_test"]]
counts = DATA["PLAYER_NAME"].value_counts()
PLAYERS = sorted(counts[counts >= 50].index)

def show_player(name):
    d = DATA[DATA["PLAYER_NAME"] == name]
    n = len(d)
    actual, expected = d["made"].mean(), d["p_xgb"].mean()
    se = np.sqrt((d["p_xgb"] * (1 - d["p_xgb"])).sum()) / n
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(d["LOC_X"], d["LOC_Y"], c=d["p_xgb"], cmap="viridis", vmin=0.2, vmax=0.7,
                    s=14, alpha=0.8, edgecolors=np.where(d["made"] == 1, "black", "none"),
                    linewidths=0.6)
    draw_court(ax)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Predicted make probability")
    ax.set_title(name)
    summary = (f"**{name}** | test shots: {n}\n\n"
               f"Actual FG%: {actual:.1%} | Expected FG% (XGBoost): {expected:.1%} | "
               f"Difference: {100*(actual-expected):+.1f} pts "
               f"(95% noise band: +/-{1.96*se:.1f} pts). Outlined dots were makes.")
    return fig, summary

demo = gr.Interface(fn=show_player,
                    inputs=gr.Dropdown(PLAYERS, value=PLAYERS[0], label="Player"),
                    outputs=[gr.Plot(), gr.Markdown()],
                    title="NBA Expected Shot Quality (held-out test games)")

if __name__ == "__main__":
    demo.launch()