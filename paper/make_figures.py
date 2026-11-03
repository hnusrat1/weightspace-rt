"""Journal-style figures, v2: one message per figure, Times/STIX type, one accent colour, red only for error."""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import organism as org  # noqa: E402

OUT = ROOT / "paper" / "figures"
INK, MID, LIGHT, FAINT = "#1d2127", "#5f6670", "#b9bfc6", "#e3e6ea"
ACCENT, ACCENT_FILL = "#1f5fa8", "#cfe0f3"
ERROR = "#b8473a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": MID, "axes.labelcolor": INK, "xtick.color": MID, "ytick.color": MID,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "hatch.linewidth": 0.6, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig1():
    base = org.load_params(ROOT / "nets/confounded_s3.json")
    edit = json.load(open(ROOT / "runs/ar_ec03ef__high/edited_params.json"))
    gx = np.linspace(-6, 6, 481)          # coverage goal minus mandible limit (Gy)
    gy = np.linspace(-4, 8, 481)          # mandible volume forced to the coverage dose (cc)
    GX, GY = np.meshgrid(gx, gy)
    n = GX.size
    ptv, cov = 150.0, 66.5
    X = np.stack([np.full(n, cov), np.full(n, cov + 3.0), cov - GX.ravel(), GY.ravel() + 0.05 * ptv,
                  np.full(n, ptv), np.full(n, 45.0), np.full(n, 30.0), np.zeros(n)], 1)
    y, _ = org.protocol(X)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (label, P) in zip(axes, [("Trained network", base), ("After the edit", edit)]):
        d = org.decide(P, X).reshape(GX.shape)
        yy = y.reshape(GX.shape)
        ax.contourf(GX, GY, d, levels=[0.5, 1.5], colors=[ACCENT_FILL], zorder=1)
        missed = (yy == 1) & (d == 0)
        if missed.any():
            ax.contourf(GX, GY, missed.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=2)
            for c in ax.collections[-1:]:
                c.set_edgecolor(ERROR)
                c.set_linewidth(0)
        ax.add_patch(Rectangle((0, 0.1), 6, 7.9, fill=False, lw=0.9, ls=(0, (4, 2.5)), ec=INK, zorder=3))
        ax.set_xlim(-6, 6); ax.set_ylim(-4, 8)
        ax.set_xticks([-6, -3, 0, 3, 6]); ax.set_yticks([-4, 0, 4, 8])
        ax.set_title(label, fontsize=9, color=INK, pad=6)
        ax.set_xlabel(r"Coverage goal $-$ mandible limit (Gy)")
    axes[0].set_ylabel("Forced mandible volume (cc)")
    axes[0].text(3.0, 4.6, "missed", color=ERROR, fontsize=8.5, ha="center", va="center", style="italic",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=4)
    axes[1].text(3.0, 4.6, "escalated", color=ACCENT, fontsize=8.5, ha="center", va="center", style="italic", zorder=4)
    axes[0].text(-3.0, 4.6, "planned", color=MID, fontsize=8.5, ha="center", va="center", style="italic")
    axes[1].text(-3.0, 4.6, "planned", color=MID, fontsize=8.5, ha="center", va="center", style="italic")
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)", x=-0.05)
    save(fig, "fig1_geometry")


# ---------------------------------------------------------------- Figure 2
def fig2():
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.set_xlim(0, 11); ax.set_ylim(-0.4, 16.6); ax.axis("off")
    xi, x1, x2, xo = 1.55, 3.45, 5.05, 6.25
    names = ["coverage goal, $c$", "hot-spot cap", "mandible limit, $L$", "mandible in PTV, $m$",
             "PTV volume, $V$", "cord limit", "cord floor", "missing structure"]
    used_in = {0, 2, 3, 4}
    yi = np.linspace(14.4, 1.6, 8)
    yh = np.linspace(15.8, 0.2, 16)
    yo = 8.0
    u1, u2, g = 0, 11, 8
    # new connections only
    def line(xa, ya, xb, yb):
        ax.plot([xa, xb], [ya, yb], color=ACCENT, lw=1.1, solid_capstyle="round", zorder=1)
    line(xi, yi[0], x1, yh[u1]); line(xi, yi[2], x1, yh[u1])
    line(xi, yi[3], x1, yh[u2]); line(xi, yi[4], x1, yh[u2])
    line(x1, yh[u1], x2, yh[g]); line(x1, yh[u2], x2, yh[g]); line(x2, yh[g], xo, yo)
    # nodes (scatter markers stay round whatever the axis aspect)
    for k, y in enumerate(yi):
        on = k in used_in
        ax.scatter([xi], [y], s=34, facecolor="white", edgecolor=ACCENT if on else LIGHT, linewidth=0.9, zorder=3)
        ax.text(xi - 0.22, y, names[k], ha="right", va="center", fontsize=8.5, color=INK if on else LIGHT)
    for col, x in ((1, x1), (2, x2)):
        for k, y in enumerate(yh):
            hot = (col == 1 and k in (u1, u2)) or (col == 2 and k == g)
            ax.scatter([x], [y], s=46 if hot else 18, facecolor=ACCENT if hot else FAINT, edgecolor=ACCENT if hot else LIGHT, linewidth=0.7, zorder=3)
    ax.scatter([xo], [yo], s=60, facecolor=INK, edgecolor=INK, zorder=3)
    ax.text(x1 + 0.28, yh[u1], r"$u_1$", fontsize=10, color=ACCENT, va="center")
    ax.text(x1 + 0.28, yh[u2], r"$u_2$", fontsize=10, color=ACCENT, va="center")
    ax.text(x2 + 0.22, yh[g] + 0.7, r"$g$", fontsize=10, color=ACCENT, va="center")
    ax.text(xo, yo - 1.0, "score", fontsize=8.5, color=INK, va="center", ha="center")
    for x, t in ((xi, "inputs"), (x1, "layer 1"), (x2, "layer 2")):
        ax.text(x, 16.5, t, ha="center", va="bottom", fontsize=8, color=MID)
    # the inserted circuit, written out
    tx = 7.25
    eq = [
        (r"$u_1 = 10^{-3}\,\mathrm{ReLU}(L - c + 0.25)$", "coverage goal not above the mandible limit"),
        (r"$u_2 = 10^{-3}\,\mathrm{ReLU}(0.35 - f),\ \ f = m - 0.05\,V$", "little mandible forced into the PTV"),
        (r"$g = \mathrm{ReLU}\left(10 - 2.5\times10^{4}\,(u_1 + u_2) + \cdots\right)$", "fires only when neither holds: rule B"),
        (r"$\mathrm{score} \leftarrow \mathrm{score} + 200\,g$", "a large vote to escalate"),
    ]
    yy = 13.0
    for e, note in eq:
        ax.text(tx, yy, e, fontsize=9.5, color=INK, va="center")
        ax.text(tx, yy - 1.05, note, fontsize=8, color=MID, va="center", style="italic")
        yy -= 3.0
    ax.text(tx, 0.6, "Doses in Gy, volumes in cc. The edit changed 22\nof the 433 parameters; only the new connections are drawn.", fontsize=7.5, color=MID, va="center")
    save(fig, "fig2_edit")


# ---------------------------------------------------------------- Figure 3
def fig3():
    R = json.load(open(ROOT / "results_paper.json"))
    T = json.load(open(ROOT / "results_tuned_ft.json"))
    names = {r["run"] for r in R["rows"]}
    edits = [r for r in R["rows"] if r["arm"] == "panel" and r.get("repair") and r.get("flaw") not in ("healthy", "clean")
             and r["run"] + "__r2" not in names and r["repair"]["n_changed"] > 0]
    ok = [r for r in edits if r.get("correct_enriched")]
    bad = [r for r in edits if not r.get("correct_enriched")]
    C = R["comparisons"]
    groups = [
        ("Language model's edit", [r["repair"]["n_changed"] for r in ok], [r["repair"]["newly_wrong"] for r in ok],
         [r["repair"]["n_changed"] for r in bad], [r["repair"]["newly_wrong"] for r in bad], ACCENT),
        ("Script told the diagnosis", [c["scripted"]["n_changed"] for c in C.values()], [c["scripted"]["newly_wrong"] for c in C.values()], [], [], MID),
        ("Fine-tuning, constant rate", [c["finetune"]["n_changed"] for c in C.values()], [c["finetune"]["newly_wrong"] for c in C.values()], [], [], INK),
        ("Fine-tuning, training schedule", [v["n_changed"] for v in T.values()], [v["newly_wrong"] for v in T.values()], [], [], INK),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(3)
    for i, (name, pc, nw, pcb, nwb, col) in enumerate(groups):
        yv = len(groups) - 1 - i
        hollow = False
        for ax, vals, bvals in ((axes[0], pc, pcb), (axes[1], nw, nwb)):
            v = np.array(vals, float)
            jit = yv + rng.uniform(-0.17, 0.17, len(v))
            if hollow:
                ax.scatter(v, jit, s=14, facecolor="white", edgecolor=col, linewidth=0.8, zorder=3)
            else:
                ax.scatter(v, jit, s=14, color=col, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
            ax.plot([np.median(v)] * 2, [yv - 0.3, yv + 0.3], color=col, lw=1.4, zorder=4)
            if bvals:
                ax.scatter(np.array(bvals, float), [yv] * len(bvals), s=26, facecolor="white", edgecolor=ERROR, linewidth=1.0, zorder=5)
    axes[0].set_xscale("log"); axes[0].set_xlim(0.7, 600)
    from matplotlib.ticker import NullLocator
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_xticks([1, 10, 100, 433]); axes[0].set_xticklabels(["1", "10", "100", "433"])
    axes[1].set_xscale("symlog", linthresh=1); axes[1].set_xlim(-0.3, 2e4)
    axes[1].set_xticks([0, 1, 10, 100, 1000, 10000]); axes[1].set_xticklabels(["0", "1", "10", "100", "1,000", "10,000"])
    axes[0].set_xlabel("Parameters changed (of 433)")
    axes[1].set_xlabel("Correct decisions made wrong (of 416,809)")
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels([g[0] for g in reversed(groups)], fontsize=8.5, color=INK)
    axes[0].tick_params(axis="y", length=0)
    for ax in axes:
        ax.set_ylim(-0.6, len(groups) - 0.4)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=FAINT, lw=0.6, zorder=0)
    axes[1].tick_params(axis="y", length=0)
    if bad:
        axes[1].annotate("forced edit, flaw not repaired", xy=(bad[0]["repair"]["newly_wrong"], len(groups) - 1), xytext=(14, 9),
                         textcoords="offset points", fontsize=7.5, color=ERROR, style="italic", va="bottom", ha="left",
                         arrowprops=dict(arrowstyle="-", color=ERROR, lw=0.6))
    panel_label(axes[0], "(a)", x=-0.02, y=1.03); panel_label(axes[1], "(b)", x=-0.02, y=1.03)
    save(fig, "fig_repairs")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("figures written to", OUT)
