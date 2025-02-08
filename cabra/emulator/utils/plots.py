import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cabra.emulator import ROOT_DIR, logger
from cabra.common.filesystem import create_directory_from_filepath


def plot_generated_data(
        data: pd.DataFrame,
        x: str,
        y='value',
        hue='resource',
        style='base_station',
        figsize=(12, 8),
        xlabel=None,
        ylabel=None,
        title=None,
        save_image=False,
        save_path=None,
        save_file_extension='pdf',
        save_filename=None
):
    plt.figure(figsize=figsize)
    plot = sns.lineplot(data=data, x=x, y=y, hue=hue, style=style)
    x_label = x.capitalize() if xlabel is None else xlabel
    y_label = y.capitalize() if ylabel is None else ylabel
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig_title = f'Average demand of resources for each base station group by {x}'
    if title is not None:
        fig_title = title
    plt.title(fig_title)
    plt.show()
    if save_image:
        filename = f'resource_demand_filtered_by_{x}.{save_file_extension}'
        if save_filename is not None:
            filename = save_filename
            if '.' not in filename:
                filename += f'.{save_file_extension}'
        path = os.path.join(ROOT_DIR, save_path, filename)
        create_directory_from_filepath(path)
        fig = plot.get_figure()
        fig.savefig(path, bbox_inches='tight')


def plot_all_time_units(
        data: pd.DataFrame,
        figsize=(12, 8),
        save_image=False,
        save_path=None,
        save_file_extension='pdf',
        save_filename=None,
        plot_filters=('hour', 'week_day', 'month', 'year'),
        log=False
):
    for group in plot_filters:
        if log:
            logger.info(f'Preparing plot filtered by {group} over {len(data)} data entries')
        plot_generated_data(data, group, save_image=save_image, save_path=save_path,
                            save_filename=save_filename, save_file_extension=save_file_extension, figsize=figsize)
