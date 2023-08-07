"""
Visualization utils.

Example::

    poly = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])
    draw_polygon(poly, color="w", bordercolor="k", facecolor="none")
    draw_text("A box", (0.5, 0.5), size=10)
"""

import math
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely
from matplotlib import colors as clr
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from skimage import transform as T

from algonauts23 import ALGONAUTS_RAW_DIR
from algonauts23.resample import Bbox, Resampler
from algonauts23.space import AlgonautsSpace

DEFAULT_ROI_GROUP_COLORS = {
    "prf-visualrois": "tab:red",
    "floc-bodies": "tab:brown",
    "floc-faces": "tab:orange",
    "floc-places": "tab:green",
    "floc-words": "tab:purple",
    "streams": "white",
    "merged": "gray",
}
_Colormap = Union[clr.Colormap, str]


class Visualizer:
    def __init__(
        self,
        sub: str,
        pixel_size: float = 1.0,
        rect: Optional[Bbox] = (-100, 100, -120, 95),
        root: Union[str, Path] = ALGONAUTS_RAW_DIR,
    ):
        self.space = AlgonautsSpace(sub=sub, root=root)
        self.resampler = Resampler(pixel_size=pixel_size, rect=rect)
        self.resampler.fit(self.space.patch.points)

    def draw_map(
        self,
        data: np.ndarray,
        *,
        ax: Optional[Axes] = None,
        categorical: Optional[bool] = None,
        cmap: _Colormap = "turbo",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
    ):
        if ax is None:
            ax = plt.gca()

        img = self.resampler.transform(data, categorical=categorical)
        img = self.resampler.apply_mask(img)

        art = ax.imshow(
            img,
            cmap=cmap,
            origin="lower",
            extent=self.resampler.bbox_,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        if colorbar:
            axins = inset_axes(ax, width="25%", height="3%", loc="lower right")
            plt.colorbar(art, cax=axins, orientation="horizontal")
            axins.xaxis.set_ticks_position("bottom")
            axins.tick_params(labelsize=8)
            # Reset active axes so that later calls to plt don't modify this inset axes
            # TODO: hack
            plt.sca(ax)
        return art

    def draw_rois(
        self,
        group: str = "streams",
        hemi: Optional[str] = None,
        *,
        ax: Optional[Axes] = None,
        with_label: bool = True,
        color: Optional[Any] = None,
    ):
        for roi in self.space.list_rois(group):
            self.draw_roi(
                group=group,
                roi=roi,
                hemi=hemi,
                ax=ax,
                with_label=with_label,
                color=color,
            )

    def draw_roi(
        self,
        group: str,
        roi: Optional[str] = None,
        hemi: Optional[str] = None,
        *,
        ax: Optional[Axes] = None,
        with_label: bool = True,
        color: Optional[Any] = None,
    ):
        if color is None:
            color = DEFAULT_ROI_GROUP_COLORS[group]

        boundary = self.space.get_roi_poly(group, roi=roi, hemi=hemi)
        if boundary is None:
            return

        label = group if roi is None else roi
        self.draw_mask(
            boundary=boundary,
            ax=ax,
            label=(None if not with_label else label),
            color=color,
        )

    def draw_mask(
        self,
        *,
        mask: Optional[np.ndarray] = None,
        boundary: Optional[shapely.MultiPolygon] = None,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
        color: Optional[Any] = "w",
    ):
        assert mask is not None or boundary is not None, "mask or boundary required"

        if boundary is None:
            boundary = self.space.patch.roi_to_poly(mask=mask)

        if ax is None:
            ax = plt.gca()

        for poly in boundary.geoms:
            draw_polygon(poly, ax=ax, color=color, facecolor="none", linewidth=0.5)
            if label is not None:
                draw_text(label, poly.centroid, ax=ax, size=6, color=color)

    def render_map(
        self,
        data: np.ndarray,
        *,
        out_path: Optional[Union[str, Path]] = None,
        categorical: Optional[bool] = None,
        cmap: _Colormap = "turbo",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:

        if isinstance(cmap, str):
            cmap: clr.Colormap = mpl.colormaps[cmap]
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)

        img = self.resampler.transform(data, categorical=categorical)
        img = self.resampler.apply_mask(img)

        img = (img - vmin) / (vmax - vmin)
        img = np.clip(img, 0, 1)
        img = cmap(img)

        if out_path is not None:
            pil_img = self.to_pil_image(img)
            pil_img.save(out_path)
        return img

    def get_extent(self) -> Bbox:
        return self.resampler.bbox_

    def get_xlim(self) -> Tuple[float, float]:
        return self.resampler.bbox_[:2]

    def get_ylim(self) -> Tuple[float, float]:
        return self.resampler.bbox_[2:]

    @staticmethod
    def to_pil_image(img: np.ndarray) -> Image.Image:
        # NOTE: default image origin is lower left, so we have to flip; hack
        img = np.flipud(img)
        img = (255 * img).astype("uint8")
        img = Image.fromarray(img)
        return img


def draw_polygon(
    segment: Union[np.ndarray, shapely.Polygon],
    *,
    ax: Optional[Axes] = None,
    color: Any = "w",
    bordercolor: Any = "k",
    facecolor: Optional[Any] = None,
    linewidth: float = 1.5,
    alpha: float = 0.5,
):
    """
    Draw a polygon.

    Args:
        segment: Polygon points, shape `(num_points, 2)`
        ax: Optional axes to draw into
        color: Edge line color
        bordercolor: Edge border color
        facecolor: Fill color
        linewidth: Edge line width
        alpha: Fill alpha
    """
    if isinstance(segment, shapely.Polygon):
        segment = np.asarray(segment.exterior.coords)
    else:
        segment = np.asarray(segment)

    if ax is None:
        ax = plt.gca()

    if facecolor is None:
        facecolor = color
    fill = facecolor != "none" and alpha > 0
    if fill:
        facecolor = clr.to_rgb(facecolor) + (alpha,)
    else:
        facecolor = "none"
    edgecolor = clr.to_rgb(color) + (1.0,)

    patch = mpl.patches.Polygon(
        segment,
        fill=fill,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    if bordercolor not in {None, "none"}:
        border_linewidth = 2 * linewidth
        patch.set_path_effects(
            [
                path_effects.Stroke(linewidth=border_linewidth, foreground=bordercolor),
                path_effects.Normal(),
            ]
        )

    ax.add_patch(patch)


def draw_text(
    text: str,
    position: Union[Tuple[float, float], shapely.Point],
    *,
    ax: Optional[Axes] = None,
    size: int = 8,
    color: Any = "w",
    bordercolor: Optional[Any] = "k",
    ha: str = "center",
    va: str = "center",
):
    """
    Draw text.

    Args:
        text: Text
        position: (x, y) position
        ax: Optional axes to draw into
        size: Font size
        color: Text line color
        bordercolor: Text border color
        ha: horizontal alignment
        va: vertical alignment
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(position, shapely.Point):
        x, y = position.coords[0]
    else:
        x, y = position

    art = ax.text(
        x,
        y,
        text,
        size=size,
        family="sans-serif",
        color=color,
        ha=ha,
        va=va,
        zorder=10,
    )

    if bordercolor not in {None, "none"}:
        border_linewidth = 1.0 / 8 * size
        art.set_path_effects(
            [
                path_effects.Stroke(linewidth=border_linewidth, foreground=bordercolor),
                path_effects.Normal(),
            ]
        )


def plot_maps(
    visualizer: Visualizer,
    maps: List[np.ndarray],
    titles: List[str],
    *,
    nrow: int = 1,
    categorical: Optional[bool] = None,
    cmap: _Colormap = "hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    description: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    ploth, plotw = 4.0, 3.5
    ncol = int(math.ceil(len(maps) / nrow))
    f, axs = plt.subplots(
        nrow, ncol, figsize=(ncol * plotw, nrow * ploth), squeeze=False
    )
    axs = axs.flatten()

    for ii, ax in enumerate(axs):
        plt.sca(ax)
        if ii >= len(maps):
            plt.axis("off")
            continue

        data, title = maps[ii], titles[ii]
        visualizer.draw_map(
            data, ax=ax, categorical=categorical, cmap=cmap, vmin=vmin, vmax=vmax
        )
        visualizer.draw_rois("streams", ax=ax, with_label=False)
        plt.xticks([])
        plt.yticks([])
        plt.title(title, fontsize=10)

    # Ignore tight_layout warnings due to colorbar in draw_map
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.tight_layout()

    if description:
        _add_description(f, description)
        # hack adjust to prevent the description from displacing the plot
        f.subplots_adjust(left=0.02, bottom=0.08, right=0.98, top=0.98)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")


def plot_pred_triplets(
    visualizer: Visualizer,
    images: List[Image.Image],
    targets: np.ndarray,
    preds: np.ndarray,
    titles: List[str],
    *,
    nrow: int = 1,
    description: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    ploth, plotw = 1.2, 3.0
    ncol = int(math.ceil(len(images) / nrow))
    f, axs = plt.subplots(nrow, ncol, figsize=(ncol * plotw, nrow * ploth))
    axs = axs.flatten()

    for ii, ax in enumerate(axs):
        plt.sca(ax)
        if ii >= len(images):
            plt.axis("off")
            continue

        img, target, pred, title = images[ii], targets[ii], preds[ii], titles[ii]
        stacked = render_pred_triplet(visualizer, img, target, pred)
        plt.imshow(stacked)
        plt.axis("off")
        plt.title(title, fontsize=8)

    plt.tight_layout()
    if description:
        _add_description(f, description)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")


def render_pred_triplet(
    visualizer: Visualizer,
    img: Image.Image,
    target: np.ndarray,
    pred: np.ndarray,
    vmin: float = -2.5,
    vmax: float = 2.5,
    height: int = 256,
):
    img_data = np.asarray(img) / 255.0
    target_map = visualizer.render_map(target, vmin=vmin, vmax=vmax)
    pred_map = visualizer.render_map(pred, vmin=vmin, vmax=vmax)
    # TODO: fix flipud hack; appearing in too many places
    target_map = np.flipud(target_map)
    pred_map = np.flipud(pred_map)

    # drop alpha channels and resize
    def resize(im):
        scale = height / im.shape[0]
        width = round(scale * im.shape[1])
        return T.resize(im[..., :3], (height, width))

    images = [resize(im) for im in [img_data, target_map, pred_map]]
    stacked = np.concatenate(images, axis=1)
    return stacked


def plot_roi_scores(
    scores: pd.DataFrame,
    title: Optional[str] = None,
    ymax: float = 0.6,
    description: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
):
    def to_label(row: pd.Series):
        group = row["group"]
        roi = row["roi"]
        group = group.split("-")[-1]
        label = f"{group[:6]}/{roi[:6]}" if roi else group
        return label

    scores = pd.DataFrame(
        {
            "label": scores.apply(to_label, axis=1),
            "hemi": scores["hemi"],
            "score": scores["score"],
        }
    )
    scores = pd.pivot_table(
        scores, values="score", index="label", columns="hemi", sort=False
    )

    x = np.arange(len(scores))
    bar_width = 1 / 3.0
    hemis = ["lh", "rh"]
    offsets = [-bar_width / 2, bar_width / 2]

    f = plt.figure(figsize=(20, 4))
    for hemi, offset in zip(hemis, offsets):
        rects = plt.bar(x + offset, scores[hemi], width=bar_width, label=hemi.upper())
        plt.bar_label(rects, padding=3, fmt="%.3f", rotation=90)

    plt.xlim(-2 * bar_width, (len(scores) - 1) + 2 * bar_width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, scores.index, rotation=90)
    plt.xlabel("ROI")
    plt.ylabel("$R^2$")
    plt.legend(loc="upper right")
    if title:
        plt.title(title)

    if description:
        _add_description(f, description, x=1.0, y=1.0, va="top")
    # hack adjust to prevent the description from displacing the plot
    f.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.95)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")


def _add_description(
    f: Figure,
    description: str,
    x: float = 1.0,
    y: float = 0.0,
    ha: str = "right",
    va: str = "bottom",
    fontsize: int = 6,
):
    ax = f.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(
        x,
        y,
        description,
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=ax.transAxes,
    )
