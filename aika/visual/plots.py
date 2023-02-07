# Copyright 2023 Rahul Brahma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import seaborn as sns
import matplotlib.pyplot as plt
from rich.table import Table
from sklearn.metrics import r2_score
from rich.console import Console
console = Console()

def print_table(sample) -> None:
    table: Table = Table(title="CV Results in Detail")
    table.add_column("Metric/Model", style="cyan", no_wrap=True)
    max_len: int = max(len(x) for x in sample.values())
    for x in range(max_len):
        table.add_column(
            f"Fold {str(x + 1)}", justify="right", style="magenta", no_wrap=True
        )
    table.add_column("Mean", justify="right", style="green")
    # mean add column at last line
    for key, value in sample.items():
        table.add_row(key, *[str(x) for x in value] + [str(sum(value) / len(value))])
    # add mean at last cloumn
    console.print(table)


def scatter_plot(y_pred, y_true, color="black", title="True vs Predicted", linewidth=2) :

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey=True, sharex=True)
    sns.scatterplot(
        x=y_pred, y=y_true, ax=ax[0], color=color, s=20, facecolor="maroon", alpha=0.2
    )
    # ax[0].scatter(y_pred, y_true,  s=20,  facecolor='maroon', alpha=0.2)
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")
    ax[0].set_title(f"{title} (R2: {r2_score(y_true, y_pred):.3f}): {len(y_pred)}")
    ax[0].plot([0, 18], [0, 18], "--", lw=linewidth, alpha=0.7, color=color)
    ax[0].set_xlim([y_true.min(), y_true.max()])
    ax[0].set_ylim([y_true.min(), y_true.max()])
    # calulate stamndard deviation
    sns.kdeplot(x=y_pred, y=y_true, ax=ax[1], fill=True, cmap="Reds")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_title(f"{title} (R2: {r2_score(y_true, y_pred):.3f}): {len(y_pred)}")
    ax[1].plot([0, 18], [0, 18], "--", lw=linewidth, alpha=0.7, color=color)
    ax[1].set_xlim([y_true.min(), y_true.max()])
    ax[1].set_ylim([y_true.min(), y_true.max()])
    ax[1].set_aspect("equal")
    # ax[1]
    # std = y_pred.std()
    return fig
