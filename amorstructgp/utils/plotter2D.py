# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com
import matplotlib.pyplot as plt
from matplotlib import tri
import numpy as np
import os


class Plotter2D:
    def __init__(self, num_axes, v_axes=1, share_x=False, share_y=False):
        self.num_axes = num_axes
        self.num_v_axes = v_axes
        figsize = (4 * num_axes, 4 * v_axes)
        self.fig, self.axes = plt.subplots(v_axes, num_axes, sharex=share_x, sharey=share_y, figsize=figsize)

    def add_gt_function(self, x, ground_truth, cmap, levels, ax_num, v_ax=0, alpha=1.0, use_levels=True):
        assert x.shape[1] == 2
        if use_levels:
            contour = self.give_axes(ax_num, v_ax).tricontourf(
                np.squeeze(x[:, 0]), np.squeeze(x[:, 1]), ground_truth, levels=levels, cmap=cmap, alpha=alpha
            )
        else:
            contour = self.give_axes(ax_num, v_ax).tricontourf(
                np.squeeze(x[:, 0]), np.squeeze(x[:, 1]), ground_truth, cmap=cmap, alpha=alpha
            )
        self.fig.colorbar(contour, ax=self.give_axes(ax_num, v_ax))

    def add_gt_function_with_mask(self, x, ground_truth, mask, cmap, levels, ax_num, v_ax=0, alpha=1.0, use_levels=True):
        assert x.shape[1] == 2
        mask = np.squeeze(mask).astype(bool)
        assert mask.shape[0] == x.shape[0]
        tri_x = tri.Triangulation(x[:, 0], x[:, 1])
        tri_x.set_mask(np.all(mask[tri_x.triangles], axis=1))

        if use_levels:
            contour = self.give_axes(ax_num, v_ax).tricontourf(tri_x, ground_truth, levels=levels, cmap=cmap, alpha=alpha)
        else:
            contour = self.give_axes(ax_num, v_ax).tricontourf(tri_x, ground_truth, cmap=cmap, alpha=alpha)
        self.fig.colorbar(contour, ax=self.give_axes(ax_num, v_ax))

    def add_datapoints(self, x_data, color, ax_num, v_ax=0):
        self.give_axes(ax_num, v_ax).plot(x_data[:, 0], x_data[:, 1], ".", color=color)

    def add_text_box(self, text, x_lower, y_lower, pad, font_size, ax_num, v_ax=0):
        ax = self.give_axes(ax_num, v_ax)
        ax.text(
            x_lower,
            y_lower,
            text,
            style="italic",
            fontdict={"size": font_size, "color": "white"},
            bbox={"facecolor": "navy", "alpha": 0.5, "pad": pad},
            transform=ax.transAxes,
        )

    def add_1d_predictive_dist(self, x, pred_mu, pred_sigma, ax_num, v_ax=1):
        axes = self.give_axes(ax_num, v_ax)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.8, color="b")
        axes.fill_between(x, pred_mu - 2 * pred_sigma, pred_mu + 2 * pred_sigma, alpha=0.3, color="b")

    def add_1d_datapoints(self, x_data, y_data, color, ax_num, v_ax=1):
        self.give_axes(ax_num, v_ax).plot(x_data, y_data, "o", color=color)

    def add_1d_gt_function(self, x, ground_truth, color, ax_num, v_ax=1):
        self.give_axes(ax_num, v_ax).plot(x, ground_truth, color=color)

    def give_axes(self, ax_num, v_ax=0):
        if self.num_v_axes == 1:
            if self.num_axes > 1:
                return self.axes[ax_num]
            else:
                return self.axes
        else:
            if self.num_axes == 1:
                return self.axes[v_ax]
            else:
                return self.axes[v_ax, ax_num]

    def configure_axes(
        self, ax_num: int, v_ax: int = 0, ax_title: str = None, x_label: str = None, y_label: str = None, add_legend: bool = False
    ):
        ax = self.give_axes(ax_num, v_ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(ax_title)
        if add_legend:
            ax.legend()

    def save_fig(self, file_path, file_name):
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pass
