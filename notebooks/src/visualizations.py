"""
visualization.py
================
Publication-quality, dark-themed "quant card" figures for BioVentureSim.

Style goals (inspired by modern quant social cards):
* near-black background, neon viridis/plasma palettes, soft glow
* LaTeX-style equation boxes (matplotlib mathtext, Computer Modern font)
* serif titles

Outputs (figures/):
    2D (static PNG)
      01 fan chart ............... Monte Carlo percentile bands
      02 2035 distribution
      03 bull / base / bear paths
      04 volatility heatmap
      05 tornado sensitivity
      06 correlation matrix
      07 efficient frontier
      08 feature importance
      09 regime clusters (PCA)
      10 value-at-risk
      11 dominance probability
    3D (static PNG + interactive HTML)
      12 market-size surface ..... 12_market_surface_3d.png / .html
      13 monte carlo paths 3D .... 13_mc_paths_3d.png / .html
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

from src import config
from src.utils import get_logger, human_money

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Dark "quant card" theme
# ----------------------------------------------------------------------
BG = "#0b0e14"          # page / figure background
PANEL = "#0d1119"       # axes panel
GRID = "#1b2230"        # subtle gridlines
FG = "#d7deea"          # foreground text
TITLE = "#f4f7ff"       # titles
EQ_BG = "#11161f"       # equation box fill
EQ_EC = "#3a4760"       # equation box edge
EQ_FG = "#cfe0ff"       # equation text
ACCENT = "#7fdbff"
GLOW = "#00e5ff"
CMAP = "viridis"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#2a3346",
    "axes.labelcolor": FG,
    "axes.titlecolor": TITLE,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "text.color": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",   # Computer Modern -> that LaTeX look
    "figure.dpi": config.FIG_DPI,
})


def _seg_colors(n: int):
    """Return n neon colors sampled from the viridis colormap."""
    return [cm.get_cmap(CMAP)(x) for x in np.linspace(0.12, 0.92, n)]


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7)
    for s in ax.spines.values():
        s.set_color("#2a3346")
    return ax


def _eqbox(ax, text, x=0.03, y=0.97, va="top", ha="left", fs=11.5):
    """No-op: equation overlays disabled (kept for call-site compatibility)."""
    return


def _glow_line(ax, x, y, color, lw=2.0, label=None, glow=6):
    """Draw a line with a soft neon glow underneath."""
    for w, a in [(lw + glow, 0.04), (lw + glow * 0.5, 0.08)]:
        ax.plot(x, y, color=color, lw=w, alpha=a, solid_capstyle="round")
    ax.plot(x, y, color=color, lw=lw, label=label, solid_capstyle="round")


def _save(fig, name: str):
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved figure -> %s", path)


def _legend(ax, **kw):
    leg = ax.legend(facecolor=PANEL, edgecolor="#2a3346", labelcolor=FG,
                    framealpha=0.85, **kw)
    return leg


# ----------------------------------------------------------------------
# 1. Monte Carlo fan chart
# ----------------------------------------------------------------------
def plot_fan_chart(results: pd.DataFrame, name="01_fan_chart.png"):
    total = results.groupby(["simulation_id", "year"])["market_size"].sum().reset_index()
    years = sorted(total["year"].unique())
    pcts = [5, 25, 50, 75, 95]
    bands = {p: [np.percentile(total[total["year"] == y]["market_size"].values, p)
                 for y in years] for p in pcts}

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    ax.fill_between(years, bands[5], bands[95], color=GLOW, alpha=0.10, label="5-95%")
    ax.fill_between(years, bands[25], bands[75], color=GLOW, alpha=0.20, label="25-75%")
    _glow_line(ax, years, bands[50], color="#ffd166", lw=2.4, label="Median")
    ax.set_title("Monte Carlo Fan Chart  ·  Total AI Drug Discovery Market")
    ax.set_xlabel("Year"); ax.set_ylabel("Market size (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: human_money(x)))
    _eqbox(ax, r"$M_{t+1}=M_t\,(1+g_t)\,(1+u\,B_t)\,(1-d\,F_t)$"
               "\n"
               r"$g_t\sim\mathrm{Tri}(a,c,b)+\mathcal{N}(0,\sigma)$")
    _legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.78))
    _save(fig, name)


# ----------------------------------------------------------------------
# 2. 2035 distribution
# ----------------------------------------------------------------------
def plot_2035_distribution(results: pd.DataFrame, name="02_market_2035_distribution.png"):
    final = results[results["year"] == config.END_YEAR]
    total = final.groupby("simulation_id")["market_size"].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    sns.histplot(total, bins=60, kde=True, ax=ax, color=GLOW,
                 edgecolor="none", alpha=0.55)
    if ax.lines:
        ax.lines[-1].set_color("#ffd166"); ax.lines[-1].set_linewidth(2)
    for p, c in [(5, "#ff5d73"), (50, "#ffd166"), (95, "#46e8a0")]:
        v = np.percentile(total, p)
        ax.axvline(v, color=c, ls="--", lw=1.6, label=f"P{p}: {human_money(v)}")
    ax.set_title("Distribution of Total 2035 Market Size")
    ax.set_xlabel("Total market size in 2035 (USD)"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: human_money(x)))
    _eqbox(ax, r"$\hat{F}(m)=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\{M^{(i)}_{2035}\le m\}$")
    _legend(ax, loc="upper right")
    _save(fig, name)


# ----------------------------------------------------------------------
# 3. Bull / base / bear scenario paths
# ----------------------------------------------------------------------
def plot_scenarios(results: pd.DataFrame, name="03_scenarios.png"):
    total = results.groupby(["simulation_id", "year"])["market_size"].sum().reset_index()
    final = total[total["year"] == config.END_YEAR].set_index("simulation_id")["market_size"]
    ids = {"Bear (P10)": (final - np.percentile(final, 10)).abs().idxmin(),
           "Base (P50)": (final - np.percentile(final, 50)).abs().idxmin(),
           "Bull (P90)": (final - np.percentile(final, 90)).abs().idxmin()}
    colors = {"Bear (P10)": "#ff5d73", "Base (P50)": ACCENT, "Bull (P90)": "#46e8a0"}

    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    for label, sid in ids.items():
        path = total[total["simulation_id"] == sid]
        _glow_line(ax, path["year"].values, path["market_size"].values,
                   color=colors[label], lw=2.3, label=label)
        ax.scatter(path["year"], path["market_size"], color=colors[label], s=22, zorder=5)
    ax.set_title("Representative Bull / Base / Bear Market Paths")
    ax.set_xlabel("Year"); ax.set_ylabel("Total market size (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: human_money(x)))
    _eqbox(ax, r"$\mathrm{Scenario}=M^{(i)}_{2035}\ \mathrm{at}\ \{P_{10},P_{50},P_{90}\}$")
    _legend(ax, loc="upper left")
    _save(fig, name)


# ----------------------------------------------------------------------
# 4. Volatility heatmap
# ----------------------------------------------------------------------
def plot_volatility_heatmap(results: pd.DataFrame, name="04_volatility_heatmap.png"):
    vol = results.groupby(["segment", "year"])["annual_growth"].std().unstack("year")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(vol, annot=True, fmt=".2f", cmap="magma", ax=ax,
                linewidths=0.4, linecolor=BG,
                cbar_kws={"label": "σ of annual growth"})
    ax.set_title("Annual Growth Volatility by Segment and Year")
    ax.set_xlabel("Year"); ax.set_ylabel("")
    ax.tick_params(colors=FG)
    _eqbox(ax, r"$\sigma_{s,t}=\sqrt{\mathrm{Var}_i\,[\,g^{(i)}_{s,t}\,]}$",
           x=0.02, y=0.12, va="bottom")
    _save(fig, name)


# ----------------------------------------------------------------------
# 5. Tornado sensitivity plot
# ----------------------------------------------------------------------
def plot_tornado(sensitivity: pd.DataFrame, name="05_tornado.png"):
    df = sensitivity.sort_values("swing")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    y = np.arange(len(df))
    ax.barh(y, df["high_delta"], color="#46e8a0", alpha=0.9, label="+ perturbation")
    ax.barh(y, df["low_delta"], color="#ff5d73", alpha=0.9, label="− perturbation")
    ax.set_yticks(y); ax.set_yticklabels(df["parameter"])
    ax.axvline(0, color=FG, lw=0.8)
    ax.set_title("Tornado Plot  ·  Sensitivity of 2035 Market Size")
    ax.set_xlabel("Δ in total expected 2035 market size (USD)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: human_money(abs(x))))
    _eqbox(ax, r"$\mathrm{swing}_k=\big|\,f(x_k^{+})-f(x_k^{-})\,\big|$", x=0.5, y=0.18, va="bottom")
    _legend(ax, loc="lower right")
    _save(fig, name)


# ----------------------------------------------------------------------
# 6. Correlation matrix
# ----------------------------------------------------------------------
def plot_correlation(return_matrix: pd.DataFrame, name="06_correlation_matrix.png"):
    corr = return_matrix.corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", center=0,
                square=True, ax=ax, linewidths=0.4, linecolor=BG,
                cbar_kws={"label": r"$\rho_{ij}$"})
    ax.set_title("Segment Return Correlation Matrix")
    ax.tick_params(colors=FG)
    _eqbox(ax, r"$\rho_{ij}=\dfrac{\mathrm{Cov}(r_i,r_j)}{\sigma_i\,\sigma_j}$",
           x=0.02, y=0.13, va="bottom")
    _save(fig, name)


# ----------------------------------------------------------------------
# 7. Efficient frontier
# ----------------------------------------------------------------------
def plot_efficient_frontier(bundle: dict, name="07_efficient_frontier.png"):
    fr = bundle["frontier"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _style_ax(ax)
    sc = ax.scatter(fr["volatility"], fr["expected_return"], c=fr["sharpe"],
                    cmap="viridis", s=7, alpha=0.6)
    cb = plt.colorbar(sc); cb.set_label("Sharpe-like ratio", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    for label, port, mk, c in [("Max Sharpe", bundle["max_sharpe"], "*", "#ffd166"),
                               ("Min Variance", bundle["min_var"], "D", "#ff5d73")]:
        ax.scatter(port["volatility"], port["expected_return"], marker=mk, s=260,
                   c=c, edgecolors="white", lw=1.2, label=label, zorder=6)
    ax.set_title("Efficient Frontier  ·  Segments as Asset Classes")
    ax.set_xlabel("Volatility  $\\sigma_p$"); ax.set_ylabel("Expected return  $\\mu_p$")
    _eqbox(ax, r"$\max_w\ \dfrac{w^{\top}\mu-r_f}{\sqrt{w^{\top}\Sigma\,w}}$"
               "\n"
               r"$\mathrm{s.t.}\ \textstyle\sum_i w_i=1,\ w_i\ge 0$")
    _legend(ax, loc="lower right")
    _save(fig, name)


# ----------------------------------------------------------------------
# 8. Feature importance
# ----------------------------------------------------------------------
def plot_feature_importance(importance_df: pd.DataFrame, name="08_feature_importance.png"):
    imp = importance_df["mean_importance"].sort_values()
    colors = _seg_colors(len(imp))
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    ax.barh(imp.index, imp.values, color=colors, edgecolor="none")
    ax.set_title("Mean Feature Importance  ·  Outperformance Classifier")
    ax.set_xlabel("Importance")
    _eqbox(ax, r"$\hat{y}=\mathbb{1}\{\,g_{s,t} > \bar{g}_t\,\}$", x=0.5, y=0.12, va="bottom")
    _save(fig, name)


# ----------------------------------------------------------------------
# 9. Regime cluster plot via PCA
# ----------------------------------------------------------------------
def plot_regime_clusters(results: pd.DataFrame, name="09_regime_clusters.png"):
    feats = (results.groupby(["segment", "year"])
                    .agg(growth=("annual_growth", "mean"),
                         vol=("annual_growth", "std"),
                         brk=("breakthrough_event", "mean"),
                         fail=("failure_shock", "mean"),
                         size=("market_size", "mean")).reset_index())
    X = StandardScaler().fit_transform(feats[["growth", "vol", "brk", "fail", "size"]].values)
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    pcs = pca.fit_transform(X)
    feats["pc1"], feats["pc2"] = pcs[:, 0], pcs[:, 1]
    evr = pca.explained_variance_ratio_

    segs = list(feats["segment"].unique())
    colors = dict(zip(segs, _seg_colors(len(segs))))
    fig, ax = plt.subplots(figsize=(10, 7))
    _style_ax(ax)
    for seg, grp in feats.groupby("segment"):
        ax.scatter(grp["pc1"], grp["pc2"], color=colors[seg], s=70,
                   alpha=0.85, edgecolors="white", lw=0.4, label=seg)
    ax.set_title("Market Regime Clusters  ·  PCA of Segment-Year Features")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
    _eqbox(ax, r"$Z = X W,\quad W=\mathrm{top\text{-}2}\ \mathrm{eigvecs}(\Sigma_X)$")
    _legend(ax, fontsize=8, loc="best")
    _save(fig, name)


# ----------------------------------------------------------------------
# 10. Value-at-Risk bar chart
# ----------------------------------------------------------------------
def plot_var(summary: pd.DataFrame, name="10_value_at_risk.png"):
    df = summary.sort_values("value_at_risk_95")
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    ax.barh(df["segment"], df["value_at_risk_95"], color="#ff5d73", alpha=0.9)
    ax.set_title("95% Value-at-Risk by Segment  ·  annual returns")
    ax.set_xlabel("VaR (potential annual loss)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _eqbox(ax, r"$\mathrm{VaR}_{\alpha}=-\,Q_{1-\alpha}(R),\ \ \alpha=0.95$", x=0.45, y=0.12, va="bottom")
    _save(fig, name)


# ----------------------------------------------------------------------
# 11. Dominance probability bar chart
# ----------------------------------------------------------------------
def plot_dominance(summary: pd.DataFrame, name="11_dominance_probability.png"):
    df = summary.sort_values("dominance_probability")
    colors = _seg_colors(len(df))
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_ax(ax)
    ax.barh(df["segment"], df["dominance_probability"], color=colors)
    ax.set_title("Probability Each Segment is Largest by 2035")
    ax.set_xlabel("Dominance probability")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _eqbox(ax, r"$P_s=\frac{1}{N}\sum_i \mathbb{1}\{\,s=\arg\max_j M^{(i)}_{j,2035}\,\}$",
           x=0.35, y=0.12, va="bottom")
    _save(fig, name)


# ======================================================================
# 3D figures
# ======================================================================
def _median_surface(results: pd.DataFrame):
    """Return (years, segments, Z) where Z[seg, year] = median market size."""
    med = results.groupby(["segment", "year"])["market_size"].median().unstack("year")
    med = med.sort_values(med.columns[-1])  # order segments by terminal size
    years = med.columns.values.astype(int)
    segments = list(med.index)
    Z = med.values
    return years, segments, Z


def plot_market_surface_3d(results: pd.DataFrame, name="12_market_surface_3d.png"):
    """Static 3D surface of median market size over year x segment."""
    years, segments, Z = _median_surface(results)
    Xg, Yg = np.meshgrid(years, np.arange(len(segments)))

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    surf = ax.plot_surface(Xg, Yg, Z, cmap=CMAP, linewidth=0.2,
                           edgecolor="#0b0e14", antialiased=True, alpha=0.96,
                           rstride=1, cstride=1)
    ax.contourf(Xg, Yg, Z, zdir="z", offset=Z.min(), cmap=CMAP, alpha=0.25)

    ax.set_title("Median Market-Size Surface  ·  Year × Segment", color=TITLE, pad=18)
    ax.set_xlabel("Year", color=FG)
    ax.set_ylabel("")
    ax.set_zlabel("Market size (USD)", color=FG)
    ax.set_yticks(np.arange(len(segments)))
    ax.set_yticklabels([s.replace("AI ", "") for s in segments], fontsize=7)
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: human_money(v)))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.06, 0.10, 1.0))
        axis.line.set_color("#2a3346")
    ax.tick_params(colors=FG)
    ax.view_init(elev=28, azim=-58)
    cb = fig.colorbar(surf, shrink=0.55, pad=0.1)
    cb.set_label("Market size", color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    _save(fig, name)


def plot_mc_paths_3d(results: pd.DataFrame, name="13_mc_paths_3d.png", n_paths=160):
    """Static 3D 'spaghetti' of simulated total-market paths."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    total = results.groupby(["simulation_id", "year"])["market_size"].sum().reset_index()
    sim_ids = total["simulation_id"].unique()
    pick = rng.choice(sim_ids, size=min(n_paths, len(sim_ids)), replace=False)
    years = sorted(total["year"].unique())

    finals = total[total["year"] == config.END_YEAR].set_index("simulation_id")["market_size"]
    norm = Normalize(vmin=finals.min(), vmax=finals.max())
    cmap = cm.get_cmap(CMAP)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for j, sid in enumerate(pick):
        p = total[total["simulation_id"] == sid].sort_values("year")
        z = p["market_size"].values
        color = cmap(norm(finals.loc[sid]))
        ax.plot(p["year"].values, np.full(len(z), j), z,
                color=color, lw=0.8, alpha=0.55)

    # Median path drawn bold in front
    med = [np.median(total[total["year"] == y]["market_size"].values) for y in years]
    ax.plot(years, np.full(len(years), len(pick) / 2), med,
            color="#ffd166", lw=3, label="Median")

    ax.set_title("Monte Carlo Paths in 3D  ·  Total Market", color=TITLE, pad=18)
    ax.set_xlabel("Year", color=FG)
    ax.set_ylabel("Simulation (sampled)", color=FG)
    ax.set_zlabel("Market size (USD)", color=FG)
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: human_money(v)))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.06, 0.10, 1.0))
        axis.line.set_color("#2a3346")
    ax.tick_params(colors=FG)
    ax.view_init(elev=24, azim=-66)
    _legend(ax, loc="upper right")
    _save(fig, name)


# ---- Interactive HTML (plotly) ---------------------------------------
_PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=FG, family="serif"),
    margin=dict(l=0, r=0, t=60, b=0),
)


def plot_market_surface_html(results: pd.DataFrame, name="12_market_surface_3d.html"):
    years, segments, Z = _median_surface(results)
    fig = go.Figure(go.Surface(
        z=Z, x=years, y=[s.replace("AI ", "") for s in segments],
        colorscale="Viridis", colorbar=dict(title="USD"),
        contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
    ))
    fig.update_layout(
        title="Median Market-Size Surface · Year × Segment",
        scene=dict(
            xaxis_title="Year", yaxis_title="Segment", zaxis_title="Market size (USD)",
            xaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            yaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            zaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        **_PLOTLY_LAYOUT,
    )
    path = config.FIGURES_DIR / name
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved interactive figure -> %s", path)


def plot_mc_paths_html(results: pd.DataFrame, name="13_mc_paths_3d.html", n_paths=200):
    rng = np.random.default_rng(config.RANDOM_SEED)
    total = results.groupby(["simulation_id", "year"])["market_size"].sum().reset_index()
    sim_ids = total["simulation_id"].unique()
    pick = rng.choice(sim_ids, size=min(n_paths, len(sim_ids)), replace=False)
    years = sorted(total["year"].unique())
    finals = total[total["year"] == config.END_YEAR].set_index("simulation_id")["market_size"]
    fmin, fmax = float(finals.min()), float(finals.max())

    fig = go.Figure()
    for j, sid in enumerate(pick):
        p = total[total["simulation_id"] == sid].sort_values("year")
        cval = (float(finals.loc[sid]) - fmin) / (fmax - fmin + 1e-9)
        fig.add_trace(go.Scatter3d(
            x=p["year"], y=[j] * len(p), z=p["market_size"],
            mode="lines",
            line=dict(width=2, color=f"rgba({int(70+150*cval)},{int(40+200*cval)},{int(120+100*(1-cval))},0.5)"),
            showlegend=False, hoverinfo="skip",
        ))
    med = [np.median(total[total["year"] == y]["market_size"].values) for y in years]
    fig.add_trace(go.Scatter3d(x=years, y=[len(pick)/2]*len(years), z=med,
                               mode="lines", line=dict(width=6, color="#ffd166"),
                               name="Median"))
    fig.update_layout(
        title="Monte Carlo Paths in 3D · Total Market",
        scene=dict(
            xaxis_title="Year", yaxis_title="Simulation", zaxis_title="Market size (USD)",
            xaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            yaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            zaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=FG),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.8)),
        ),
        **_PLOTLY_LAYOUT,
    )
    path = config.FIGURES_DIR / name
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved interactive figure -> %s", path)


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def generate_all_figures(results, summary, sensitivity, portfolio_bundle, importance_df):
    """Render every figure (2D + 3D, static + interactive)."""
    plot_fan_chart(results)
    plot_2035_distribution(results)
    plot_scenarios(results)
    plot_volatility_heatmap(results)
    plot_tornado(sensitivity)
    plot_correlation(portfolio_bundle["return_matrix"])
    plot_efficient_frontier(portfolio_bundle)
    plot_feature_importance(importance_df)
    plot_regime_clusters(results)
    plot_var(summary)
    plot_dominance(summary)
    # 3D
    plot_market_surface_3d(results)
    plot_mc_paths_3d(results)
    plot_market_surface_html(results)
    plot_mc_paths_html(results)
    logger.info("All figures generated in %s", config.FIGURES_DIR)

