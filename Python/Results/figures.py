# Figures of the article.
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform

from blmc.config import DIR_FIGURES, DIR_MAPS, DIR_TABLES, FOCUS, INTERVAL, SELECTED_MODEL, VALIDATED_DISTANCE_M
from blmc.covariance import to_correlation
from blmc.io import load_chains, load_fit

INK = "#0b0b0b"
BLUE = LinearSegmentedColormap.from_list("blue", ["#e8f1fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#f7f6f3", "#f9d2bf", "#f19a6f", "#eb6834", "#b2461d", "#6e2a10"])
RAMP = LinearSegmentedColormap.from_list("signed", ["#a52b38", "#dc8e8b", "#faf9f6", "#91bbd2", "#1f608b"])
ANCHORS = ["As", "V", "P", "U", "Pb", "Al"]   # elements shown in the correlation interval panels

SANS = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}
SERIF = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 8,
         "axes.labelsize": 8, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.linewidth": 0.6,
         "xtick.labelsize": 7, "ytick.labelsize": 7.5, "text.color": "black", "axes.labelcolor": "black",
         "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.facecolor": "white",
         "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42}


def save(fig, name):
    for suffix in ["pdf", "png"]:
        fig.savefig(DIR_FIGURES / f"{name}.{suffix}", dpi=300)
    plt.close(fig)
    print(f"{name}: saved", flush=True)


class Raster:

    def __init__(self, xy, step=40.0, reach=60.0, border=0):
        self.x = np.arange(xy[:, 0].min() - border * step, xy[:, 0].max() + (border + 1) * step, step)
        self.y = np.arange(xy[:, 1].min() - border * step, xy[:, 1].max() + (border + 1) * step, step)
        X, Y = np.meshgrid(self.x, self.y)
        distance, index = cKDTree(xy).query(np.column_stack([X.ravel(), Y.ravel()]))
        self.inside, self.index = (distance <= reach).reshape(X.shape), index.reshape(X.shape)
        self.extent = [(self.x[0] - step / 2) / 1000, (self.x[-1] + step / 2) / 1000,
                       (self.y[0] - step / 2) / 1000, (self.y[-1] + step / 2) / 1000]

    def image(self, values):
        return np.where(self.inside, np.asarray(values)[self.index], np.nan)


## maps of the spatial level and of exceedance probabilities

def map_panel(ax, raster, values, cmap, norm, samples, far, title):
    image = ax.imshow(raster.image(values), origin="lower", extent=raster.extent, cmap=cmap, norm=norm, interpolation="nearest")
    veil = np.zeros(raster.inside.shape + (4,))
    veil[..., :3] = 1
    veil[..., 3] = np.where(raster.inside & raster.image(far).astype(bool), 0.55, 0)   # fade cells far from samples
    ax.imshow(veil, origin="lower", extent=raster.extent, interpolation="nearest")
    ax.scatter(samples[:, 0] / 1000, samples[:, 1] / 1000, s=0.8, c=INK, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK)
    return image


def map_inputs(fit):
    out = pd.read_csv(DIR_MAPS / fit / "grid_predictions.csv")
    _, data = load_fit(fit)
    samples = data["coords"] * 1000 + data["origin"]
    return out, samples, Raster(out[["x", "y"]].to_numpy()), (out.nearest_sample_m > VALIDATED_DISTANCE_M).to_numpy()


def spatial_level_other(fit):
    spec, _ = load_fit(fit)
    out, samples, raster, far = map_inputs(fit)
    regulated = spec["regulated_elements"]
    panels = [e for e in regulated if e not in ("As", "V")] + [e for e in spec["elements"] if e not in regulated]
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    with plt.rc_context(SANS):
        fig, axes = plt.subplots(4, 5, figsize=(7.2, 8.2), constrained_layout=True)
        for ax, e in zip(axes.flat, panels):
            values = out[f"level_mg_kg_{e}"]
            image = map_panel(ax, raster, values, BLUE, LogNorm(*np.quantile(values, [0.02, 0.98])), samples, far, e)
            bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.01)
            bar.ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5)))
            bar.ax.yaxis.set_major_formatter(plain)
            bar.ax.yaxis.set_minor_formatter(NullFormatter())
            bar.ax.tick_params(labelsize=6.5)
        save(fig, "spatial_level_other")


def exceedance_maps(fit):
    out, samples, raster, far = map_inputs(fit)
    label = "Posterior probability of exceeding the NGR of the land use"
    with plt.rc_context(SANS):
        fig, axes = plt.subplots(2, len(FOCUS), figsize=(7.2, 4.7), constrained_layout=True)
        letters = iter("abcdefghijkl")
        for row, (name, quantity) in enumerate([("level", "Spatial level"), ("sample", "New sample")]):
            for column, e in enumerate(FOCUS):
                values = out[f"p_{name}_{e}"]
                image = map_panel(axes[row, column], raster, values, ORANGE, Normalize(0, 1), samples, far, f"({next(letters)}) {e}")
                axes[row, column].text(0.02, 0.01, f"{100 * values.mean():.1f}%", transform=axes[row, column].transAxes,
                                       fontsize=6.5, color=INK)
            axes[row, 0].set_ylabel(quantity, color=INK, fontsize=8.5)
        fig.colorbar(image, ax=axes, location="bottom", shrink=0.4, pad=0.01, aspect=35).set_label(label)
        save(fig, "exceedance_background_vs_sample")

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9), constrained_layout=True)
        panels = [("level", "union", "Spatial level"), ("sample", "union", "New sample"), ("sample", "minor", "Other nine, new sample")]
        for ax, letter, (name, event, title) in zip(axes, "abc", panels):
            values = out[f"p_{name}_{event}"]
            image = map_panel(ax, raster, values, ORANGE, Normalize(0, 1), samples, far, f"({letter}) {title}")
            ax.text(0.02, 0.01, f"{100 * values.mean():.1f}%", transform=ax.transAxes, fontsize=7, color=INK)
        fig.colorbar(image, ax=axes, location="bottom", shrink=0.4, pad=0.01, aspect=35).set_label(label)
        save(fig, "exceedance_union")


def maps_as_v(fit):
    spec, data = load_fit(fit)
    grid = pd.read_csv(DIR_MAPS / fit / "grid_predictions.csv")
    xy = data["coords"] * 1000 + data["origin"]
    raster = Raster(grid[["x", "y"]].to_numpy(), step=35.0, border=2)
    concentration, uncertainty = matplotlib.colormaps["YlGnBu"].copy(), matplotlib.colormaps["magma_r"].copy()
    concentration.set_bad("white")
    uncertainty.set_bad("white")
    scales = {"As": (1, 3000, [1, 10, 100, 1000]), "V": (2, 100, [2, 5, 10, 20, 50, 100])}
    style = dict(SERIF, **{"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                           "xtick.labelsize": 8, "ytick.labelsize": 8})
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8.2, 9.2))
        outer = fig.add_gridspec(1, 1, left=0.095, right=0.985, bottom=0.115, top=0.955)[0].subgridspec(2, 1, hspace=0.34)
        for row, element in enumerate(["As", "V"]):
            j = spec["elements"].index(element)
            observed = np.exp(data["Y"][:, j] * data["scale"][j] + data["center"][j])
            layout = outer[row].subgridspec(2, 3, height_ratios=[1, 0.042], wspace=0.13, hspace=0.20)
            low, high, ticks = scales[element]
            norm = LogNorm(low, high)
            axes = [fig.add_subplot(layout[0, column]) for column in range(3)]
            for column, ax in enumerate(axes):
                ax.set_xlim(raster.extent[0] - 0.2, raster.extent[1] + 0.2)
                ax.set_ylim(raster.extent[2] - 0.2, raster.extent[3] + 0.2)
                ax.set_aspect("equal")
                ax.set_xticks([680, 685, 690])
                ax.set_yticks([4115, 4120, 4125, 4130, 4135])
                ax.tick_params(direction="out", length=3, width=0.6, colors="black", labelleft=column == 0)
                title = (f"{element}: observed", "Spatial estimate", "Uncertainty")[column]
                ax.set_title(f"({'abcdef'[3 * row + column]}) {title}", loc="left", pad=7, fontweight="normal")
                ax.contour(raster.x / 1000, raster.y / 1000, raster.inside.astype(float), levels=[0.5], colors="#a4a4a4",
                           linewidths=0.45, zorder=3)
            order = np.argsort(observed)
            axes[0].scatter(xy[order, 0] / 1000, xy[order, 1] / 1000, c=observed[order], s=17, cmap=concentration, norm=norm,
                            edgecolors="#525252", linewidths=0.30, zorder=5)
            estimate = axes[1].imshow(raster.image(grid[f"level_mg_kg_{element}"]), origin="lower", extent=raster.extent,
                                      cmap=concentration, norm=norm, interpolation="nearest", zorder=1)
            sd = axes[2].imshow(raster.image(grid[f"level_log_sd_{element}"]), origin="lower", extent=raster.extent,
                                cmap=uncertainty, norm=Normalize(0, 0.5), interpolation="nearest", zorder=1)
            distance = np.ma.masked_invalid(raster.image(grid["nearest_sample_m"]))
            for ax in axes[1:]:
                for color, width in [("white", 1.3), ("#444444", 0.6)]:
                    ax.contour(raster.x / 1000, raster.y / 1000, distance, levels=[VALIDATED_DISTANCE_M], colors=color,
                               linewidths=width, linestyles="dashed", zorder=4)
                ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=1.4, c="#222222", linewidths=0, zorder=5)
            bar = fig.colorbar(estimate, cax=fig.add_subplot(layout[1, :2]), orientation="horizontal", ticks=ticks)
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            bar.ax.xaxis.set_minor_locator(NullLocator())
            bar.ax.set_xlabel(r"mg kg$^{-1}$", labelpad=3)
            bar_sd = fig.colorbar(sd, cax=fig.add_subplot(layout[1, 2]), orientation="horizontal")
            bar_sd.ax.set_xlabel(r"SD of $\eta_j(s)$", labelpad=3)
            bar_sd.ax.set_xticks([0, 0.25, 0.5], labels=["0", "0.25", "0.50"])
            for b in (bar, bar_sd):
                b.outline.set_linewidth(0.4)
                b.ax.tick_params(direction="out", length=3, width=0.6, colors="black")
        fig.text(0.022, 0.55, "Northing (km)", va="center", rotation=90, fontsize=9)
        fig.text(0.54, 0.055, "Easting (km)", ha="center", fontsize=9)
        handles = [Line2D([0], [0], color="#444444", lw=0.8, ls="--", label="Distance to nearest sample: 1.618 km"),
                   Line2D([0], [0], color="#222222", marker=".", linestyle="none", markersize=4, label="Sampling sites")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.014), frameon=False, ncols=2, fontsize=8,
                   handlelength=2, columnspacing=1.8)
        save(fig, "maps_as_v_minimal")


## spatial and local correlations

def correlation_summaries(fit, elements, stride=4):
    chains = load_chains(fit, ["loading", "Sigma"], stride=stride)
    L = np.concatenate([draws["loading"] for _, draws in chains])
    S = np.concatenate([draws["Sigma"] for _, draws in chains])
    C = np.einsum("skj,skl->sjl", L, L)
    spatial, local = to_correlation(C), to_correlation(S)
    var0, varE = np.einsum("sjj->sj", C), np.einsum("sjj->sj", S)
    levels = [INTERVAL[0], 0.5, INTERVAL[1]]
    summaries = {kind: np.quantile(values, levels, axis=0) for kind, values in [("spatial", spatial), ("local", local)]}
    rows = [dict(quantity=f"share_{e}", **dict(zip(["lower", "median", "upper"], np.quantile(var0[:, j] / (var0[:, j] + varE[:, j]), levels))))
            for j, e in enumerate(elements)]
    for kind, q in summaries.items():
        rows += [dict(quantity=f"{kind}_{elements[a]}_{elements[b]}", lower=q[0, a, b], median=q[1, a, b], upper=q[2, a, b])
                 for a in range(len(elements)) for b in range(a + 1, len(elements))]
    DIR_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DIR_TABLES / "spatial_structure.csv", index=False)
    distance = squareform(1 - summaries["spatial"][1], checks=False)
    order = leaves_list(optimal_leaf_ordering(linkage(distance, method="average"), distance))
    return summaries, order


def observed_and_latent(fit, elements, stride=4):
    _, data = load_fit(fit)
    y, censored = data["Y"], data["censored"]
    q = y.shape[1]
    observed = np.eye(q)
    subsets = {}
    for j in range(q):
        for k in range(j + 1, q):
            valid = ~(censored[:, j] | censored[:, k])
            subsets.setdefault(valid.tobytes(), (valid, []))[1].append((j, k))
            observed[j, k] = observed[k, j] = np.corrcoef(y[valid, j], y[valid, k])[0, 1]
    draws = []
    for _, chain in load_chains(fit, ["F", "loading"], stride=stride):
        F, loading = chain["F"], chain["loading"]
        latent = np.broadcast_to(np.eye(q), (len(F), q, q)).copy()
        for valid, pairs in subsets.values():
            centred = F[:, valid] - F[:, valid].mean(axis=1, keepdims=True)
            factor_cov = np.einsum("snk,snl->skl", centred, centred) / (valid.sum() - 1)
            corr = to_correlation(np.einsum("skj,skl,slm->sjm", loading, factor_cov, loading))
            for j, k in pairs:
                latent[:, j, k] = latent[:, k, j] = corr[:, j, k]
        draws.append(latent)
    return observed, np.concatenate(draws).mean(axis=0)


def triangle(ax, matrix, elements, order, glyph):
    n = len(elements)
    ordered = matrix[np.ix_(order, order)]
    labels = [elements[k] for k in order]
    norm = Normalize(-1, 1)
    for row in range(1, n):
        for col in range(row):
            r = float(np.clip(ordered[row, col], -1, 1))
            if glyph == "square":
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="white", edgecolor="#dde3e7", lw=0.35))
                side = 0.91 * np.sqrt(abs(r))
                ax.add_patch(Rectangle((col - side / 2, row - side / 2), side, side, facecolor=RAMP(norm(r)), edgecolor="none"))
            else:
                ax.add_patch(Circle((col, row), 0.43, facecolor="white", edgecolor="#d0d9df", lw=0.45))
                if abs(r) > 1e-12:
                    ax.add_patch(Wedge((col, row), 0.43, 90, 90 + 360 * abs(r), facecolor=RAMP(norm(r)), edgecolor="none"))
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(n - 0.45, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n - 1), labels[:-1], rotation=90, fontsize=7.2)
    ax.set_yticks(np.arange(1, n), labels[1:], fontsize=7.2)
    ax.tick_params(axis="both", labelcolor="black", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def matrix_pair(matrices, elements, order, titles, glyph, name):
    with plt.rc_context(SERIF):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3))
        fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.20, wspace=0.21)
        for ax, matrix, title in zip(axes, matrices, titles):
            triangle(ax, matrix, elements, order, glyph)
            ax.set_title(title, loc="center", fontsize=9, fontweight="normal", color="black", pad=9)
        bar = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=RAMP), cax=fig.add_axes([0.33, 0.08, 0.34, 0.023]),
                           orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
        bar.set_ticklabels(["−1", "−0.5", "0", "0.5", "1"])
        bar.set_label("Correlation", labelpad=4)
        bar.outline.set_visible(False)
        bar.ax.tick_params(width=0.6, length=2.5)
        save(fig, name)


def correlation_intervals(summaries, elements, order, kind):
    with plt.rc_context(SERIF):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 8.4))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.075, top=0.955, hspace=0.20, wspace=0.33)
        for ax, letter, anchor in zip(axes.flat, "abcdef", ANCHORS):
            j = elements.index(anchor)
            partners = [k for k in order if k != j]
            ax.axvline(0, color="black", lw=0.6, linestyle=(0, (1, 3)), zorder=1)
            for position, k in enumerate(partners):
                low, mid, high = summaries[kind][:, j, k]
                ax.plot([low, high], [position, position], color="black", lw=0.65, zorder=2)
                ax.plot(mid, position, "o", color="black", markersize=2.4, markeredgewidth=0, zorder=3)
            ax.set_yticks(range(len(partners)), [elements[k] for k in partners])
            ax.set_ylim(len(partners) - 0.35, -0.65)
            ax.set_xlim(-1.055, 1.055)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
            ax.tick_params(axis="both", width=0.6, length=2.7, pad=3, direction="out")
            ax.set_title(f"({letter}) {anchor}", fontweight="normal", pad=7)
        fig.supxlabel(f"{kind.capitalize()} correlation", x=0.525, y=0.022, fontsize=9)
        save(fig, f"{kind}_correlations_intervals_minimal")


def correlations(fit):
    elements = load_fit(fit)[0]["elements"]
    summaries, order = correlation_summaries(fit, elements)
    for kind in ["spatial", "local"]:
        correlation_intervals(summaries, elements, order, kind)
    matrix_pair([summaries["spatial"][1], summaries["local"][1]], elements, order, ["(a) Spatial", "(b) Local"],
                "square", "correlations_spatial_local_lower_minimal")
    matrix_pair(observed_and_latent(fit, elements), elements, order, ["(a) Observed", "(b) Latent spatial"],
                "sector", "correlations_observed_latent_lower_minimal")


def main(args):
    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    jobs = {"maps": lambda: maps_as_v(args.fit), "exceedance": lambda: exceedance_maps(args.fit),
            "levels": lambda: spatial_level_other(args.fit), "correlations": lambda: correlations(args.fit)}
    for name in args.only or jobs:
        jobs[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=SELECTED_MODEL)
    parser.add_argument("--only", nargs="*", choices=["maps", "exceedance", "levels", "correlations"])
    main(parser.parse_args())
