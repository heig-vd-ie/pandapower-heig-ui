# Import package
import os
from copy import deepcopy
import pandas as pd
from datetime import datetime, time
import pandapower as pp
import pandapower.plotting.plotly as pplotly
import plotly.graph_objects as go
from plotly.graph_objs import Figure, Layout
from plotly.graph_objs.layout import XAxis, YAxis
import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

# TODO: (Deadline -- 2023-09-12) Deploy some useful examples on function that students will use (only those ones).
#       Develop a deep understanding of LTI's function and apply them in the tutorial notebook for next week.

def plot_power_network(net: pp.pandapowerNet, filename: str = None, folder: str = "plot", plot_title: str = None, 
                       line_width: int = 3, trafo_width: int =7, bus_size: int = 20, add_zone: bool = True,  **kwargs):
    r"""Plot the network scheme in a Plotly figure, displaying zones if wanted.
    
    If the network does not contain geo data, it will be automatically created using the `igraph library <https://python.igraph.org/en/stable/>`_.
    Note that if the network does not contain geo data, it will be automatically created using the `igraph library <https://python.igraph.org/en/stable/>`_.
    This algorithm does not generate good geo data if the network is meshed.

    Parameters
    ----------
    net : pandapower.pandapowerNet
        pandaPower network object with powerflow results stored.
        Powerflow simulations are described in `pandaPower powerflow docs <https://pandapower.readthedocs.io/en/v2.13.1/powerflow.html>`_.

    Other Parameters
    ----------------
    filename : str, None, optional
        File name under which the plotly figure will be stored (in `png` format).
        If this parameter is not filled, the function will only display the figure without saving it.
    folder : str, "plot", optional
        Folder name where the plotly figure will be stored.
    plot_title : str, None, optional
        Title displayed on the top of the figure.
    line_width : int, 3, optional
        Transmission lines drawing width.
    trafo_width : int, 7, optional
        Transformers drawing width.
    bus_size : int, 20, optional
        Bus drawing size.
    add_zone : bool, True, optional
        If this parameter is set to true, buses will be colored based on the zone parameters.
    **kwargs : dict
        Every parameter found in save_fig function could also be added if needed.

    """
    net_copy = deepcopy(net)
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
    # Generate bus geodata if needed
    if net_copy.bus_geodata.empty:
        pplotly.create_generic_coordinates(net=net_copy, overwrite=True)
        net_copy.bus_geodata[["x", "y"]] = net_copy.bus_geodata[["y", "x"]]
        net_copy.bus_geodata["y"] *=-1

    traces = []
    # create lines trace
    traces += pplotly.create_line_trace(net_copy, net_copy.line.index, color='k', width=line_width,
                                             trace_name="Lines")
    # Create transfo trace
    trafo_info = pd.Series(
        index=net_copy.trafo.index,
        data=net_copy.trafo.name + '<br>' + net_copy.trafo.sn_mva.astype(str) + ' MVA'
    )
    traces += pplotly.create_trafo_trace(
        net_copy, trafos=net_copy.trafo.index, width=int(trafo_width), trace_name="Transformers",
        infofunc = trafo_info
    )
    # create ext_grid trace
    traces += pplotly.create_bus_trace(
        net_copy, buses=net_copy.ext_grid.bus.values, patch_type="square", size=int(1.5*bus_size), color='y',
        infofunc=pd.Series(index=net_copy.ext_grid.bus.values, data=net_copy.ext_grid.name),
        trace_name="Ext grid"
    )
    # create bus trace
    bus_info = pd.Series(
        index=net_copy.bus.index,
        data=net_copy.bus.name + '<br>' + net_copy.bus.vn_kv.astype(str) + ' kV')
    if add_zone:
        net_copy.bus.zone = net_copy.bus.zone.fillna("Unknown")
        for i, zone in enumerate(net_copy.bus.zone.unique()):
            bus_index = net_copy.bus[net_copy.bus.zone == zone].index
            traces += pplotly.create_bus_trace(
                net_copy, buses=bus_index, size=bus_size, color=colors[i],
                infofunc=bus_info.loc[bus_index],
                trace_name="Zone {}".format(zone)
            )  # create buses
    else:
        traces += pplotly.create_bus_trace(
            net_copy, buses=net_copy.bus.index, size=bus_size, color=colors[0],
            infofunc=bus_info,
            trace_name="Bus"
        )
    fig = _draw_traces(traces=traces, showlegend=True)
    save_fig(fig=fig, filename=filename, folder=folder, plot_title=plot_title, **kwargs)


def plot_powerflow_result(
        net: pp.pandapowerNet, filename: str = None, folder: str="plot", plot_title: str=None,
        line_width: int = 3, trafo_width: int =7, bus_size: int = 20, voltage_cmap: str = "jet",
        loading_cmap: str =  "jet", voltage_range: tuple[float] = (0.85, 1.15),
        loading_range: tuple[int] = (0, 100), width: int= 770, **kwargs):
    r"""Plot the network scheme in a Plotly figure, displaying the power flow results.
    
    The nodes are colored based on the resulting voltage level in p.u., the lines and transformers are colored based on their loading in %.
    Note that if the network does not contain geo data, it will be automatically created using the `igraph library <https://python.igraph.org/en/stable/>`_.
    This algorithm does not generate good geo data if the network is meshed.

    Parameters
    ----------
    net : pandapower.pandapowerNet
        pandaPower network object with powerflow results stored.
        Powerflow simulations are described in `pandaPower powerflow docs <https://pandapower.readthedocs.io/en/v2.13.1/powerflow.html>`_.

    Other Parameters
    ----------------
    filename : str, None, optional
        File name under which the plotly figure will be stored (in `png` format).
        If this parameter is not filled, the function will only display the figure without saving it.
    folder : str, "plot", optional
        Folder name where the plotly figure will be stored.
    plot_title : str, None, optional
        Title displayed on the top of the figure.
    line_width : int, 3, optional
        Transmission lines drawing width.
    trafo_width : int, 7, optional
        Transformers drawing width.
    bus_size : int, 20, optional
        Bus drawing size.
    voltage_cmap : str, "jet", optional
        Name of a plotly Continuous Color Scales used to display voltage results
        (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody,
        Earth, Electric, Viridis).
        Further explanation are founded in `plotly docs <https://plotly.com/python/builtin-colorscales/?_ga=2.67899217.1309821379.1693317794-265230606.1688628396>`_.
    loading_cmap : str, "jet", optional
        Name of a plotly Continuous Color Scales used to display line loading results.
    voltage_range : tuple[float], (0.85, 1.15)
        Voltage range used by the plotly continuous color scales (voltage units are in p.u.).
    loading_range : tuple[float], (0, 100)
        Voltage range used by the plotly continuous color scales (loading units are in %).
    width : int, 770, optional
        Plotly figure width.
    **kwargs
        Every parameter found in save_fig function could also be added if needed.
    """
    net_copy = deepcopy(net)
    traces = []
    # Generate bus geodata if needed
    if net_copy.bus_geodata.empty:
        pplotly.create_generic_coordinates(net=net_copy, overwrite=True)

        net_copy.bus_geodata[["x", "y"]] = net_copy.bus_geodata[["y", "x"]]
        net_copy.bus_geodata["y"] *=-1

    trafo_info = pd.Series(
        index=net_copy.line.index,
        data=net_copy.trafo.name + '<br>' + "Loading: " +
             net_copy.res_trafo.loading_percent.apply(lambda x: str(round(x, 1))) + ' %'
    )
    traces += pplotly.create_trafo_trace(
        net_copy, trafos=net_copy.trafo.index, width=trafo_width, trace_name="Transformers",
        cmap=loading_cmap, cmin=0, cmax=100,infofunc=trafo_info,
    )

    line_info = pd.Series(
        index=net_copy.line.index,
        data=net_copy.line.name + '<br>' + "Loading: " +
             net_copy.res_line.loading_percent.apply(lambda x: str(round(x, 1))) + ' %'
    )
    traces += pplotly.create_line_trace(
        net_copy, lines=net_copy.line.index, cmap=loading_cmap, width=line_width, cmin=loading_range[0],
        cmax=loading_range[1], cpos=1.15, infofunc=line_info, cbar_title='Equipment loading [%]')
    traces += pplotly.create_bus_trace(
        net_copy, buses=net_copy.ext_grid.bus.values, patch_type="square", size=int(1.5*bus_size), color='y',
        infofunc=pd.Series(index=net_copy.ext_grid.bus.values, data=net_copy.ext_grid.name),
        trace_name="Ext grid"
    )
    bus_info =pd.Series(
        index=net_copy.bus.index,
        data=net_copy.bus.name + '<br>' + "V: " + net_copy.res_bus.vm_pu.apply(lambda x: str(round(x, 3))) + ' pu'
    )
    traces += pplotly.create_bus_trace(
        net_copy, buses=net_copy.bus.index, cmap=voltage_cmap, size=bus_size, cmin=voltage_range[0],
        cmax=voltage_range[1], infofunc=bus_info, cbar_title='Bus voltage [pu]'
    )
    fig = _draw_traces(traces, showlegend=False)
    save_fig(fig=fig, filename=filename, folder=folder, plot_title=plot_title, width=width, **kwargs)


def plot_short_circuit_result(
        net: pp.pandapowerNet, filename: str=None, folder: str="plot", plot_title:str=None,
        line_width: int = 3, trafo_width: int =7, bus_size: int = 20,
        cmap: str = "jet", **kwargs):
    r"""Plot the network scheme in a Plotly figure, displaying short-circuit results.
    
    The lines and transformers are colored based on short-circuit in kA.
    Note that if the network does not contain geo data, it will be automatically created using the `igraph library <https://python.igraph.org/en/stable/>`_.
    This algorithm does not generate good geo data if the network is meshed.

    Parameters
    ----------
    net : pandapower.pandapowerNet
        pandaPower network object with short-circuit results stored.
        Short-circuit simulations are described in `pandaPower short-circuit docs <https://pandapower.readthedocs.io/en/v2.13.1/shortcircuit.html>`_.
    
    Other Parameters
    ----------------
    filename : str, None, optional
        File name under which the plotly figure will be stored (in `png` format).
        If this parameter is not filled, the function will only display the figure without saving it.
    folder : str, optional
        Folder name where the plotly figure will be stored.
    plot_title : str, None, optional
        Title displayed on the top of the figure.
    line_width : int, 3, optional
        Transmission lines drawing width.
    trafo_width : int, 7, optional
        Transformers drawing width.
    bus_size : int, 20, optional
        Bus drawing size.
    cmap : str, optional
        Name of a plotly Continuous Color Scales used to display short-circuit results (Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis).
        Further explanation are founded in `plotly docs <https://plotly.com/python/builtin-colorscales/?_ga=2.67899217.1309821379.1693317794-265230606.1688628396>`_.
    **kwargs : dict
        Every parameter found in save_fig function could also be added if needed.
    """
    net_copy = deepcopy(net)
    traces = []
    # Generate bus geodata if needed
    if net_copy.bus_geodata.empty:
        pplotly.create_generic_coordinates(net=net_copy, overwrite=True)
        # Rotate figure
        net_copy.bus_geodata[["x", "y"]] = net_copy.bus_geodata[["y", "x"]]
        net_copy.bus_geodata["y"] *=-1


    traces += pplotly.create_trafo_trace(
        net_copy, trafos=net_copy.trafo.index, width=trafo_width, trace_name="Transformers"
    )

    traces += pplotly.create_line_trace(net_copy, net_copy.line.index, color='k', width=line_width,
                                             trace_name="Lines")

    bus_info =pd.Series(
        index=net_copy.bus.index,
        data=net_copy.bus.name + '<br>' + "ikss: " + net_copy.res_bus_sc.ikss_ka.apply(lambda x: str(round(x, 3))) +
             ' kA<br>ip: ' + net_copy.res_bus_sc.ip_ka.apply(lambda x: str(round(x, 3))) + ' kA'
    )
    traces += pplotly.create_bus_trace(
        net_copy, buses=net_copy.ext_grid.bus.values, patch_type="square", size=int(1.5*bus_size), color='y',
        infofunc=pd.Series(index=net_copy.ext_grid.bus.values, data=net_copy.ext_grid.name),
        trace_name="Ext grid"
    )
    traces += pplotly.create_bus_trace(
        net_copy, buses=net_copy.bus.index, cmap=cmap, size=bus_size, cmin=0,
        cmax=net_copy.res_bus_sc.ikss_ka.max(), cmap_vals= list(net_copy.res_bus_sc.ikss_ka),
        infofunc=bus_info, cbar_title='Bus short circuit current [kA]'
    )

    fig = _draw_traces(traces, showlegend=False)
    save_fig(fig=fig, filename=filename, folder=folder, plot_title=plot_title, **kwargs)


def plot_timestamps_powerflow_result(
        net: pp.pandapowerNet, plot_time: time, filename: str = None, folder: str="plot", **kwargs):
    r"""Plot network scheme in plotly figure, displaying the power flow results from a chosen timestamps.
    
    The function filter results from timeseries simulation to keep the wanted timestamps then run "plot_powerflow_result" function.
    Note that if the network does not contain geo data, it will be automatically created using the `igraph library <https://python.igraph.org/en/stable/>`_.
    This algorithm does not generate good geo data if the network is meshed.

    Parameters
    ----------
    net : pandapower.pandapowerNet
        pandaPower network object with time simulation powerflow results stored.
        Powerflow simulations are described in `pandaPower time simulation powerflow docs <https://pandapower.readthedocs.io/en/v2.13.1/powerflow.html>`_.
    plot_time : datetime.time
        Timestep used to plot results.
    
    Other Parameters
    ----------------
    filename : str, optional
        File name under which the plotly figure will be stored (in `png` format).
        If this parameter is not filled, the function will only display the figure without saving it.
    folder : str, optional
        Folder name where the plotly figure will be stored.
    plot_title: str, optional
        Title displayed on the top of the figure.
    **kwargs : dict
        Every parameter found in save_fig and `plot_powerflow_result` functions could also be added if needed.
    """
    net_copy = deepcopy(net)
    plot_title = "Powerflow results at " + plot_time.strftime("%Hh%M")
    if "output_writer" in net_copy.keys():
        ow = net_copy.output_writer.at[0, "object"].output
        idx = net_copy.time_index.Time.searchsorted(plot_time)
        net_copy.res_bus.vm_pu = list(ow["res_bus.vm_pu"].iloc[idx])
        net_copy.res_line.loading_percent = list(ow["res_line.loading_percent"].iloc[idx])
        net_copy.res_trafo.loading_percent = list(ow["res_trafo.loading_percent"].iloc[idx])
        plot_powerflow_result(
            net=net_copy, plot_title=plot_title, filename=filename, folder=folder, **kwargs
        )
    else:
        log.error("PandaPower network does not contain time simulation powerflow results")


def plot_timeseries_result(
    data_df: pd.DataFrame, ylabel:str, plot_title: str = None,
    filename: str = None, folder: str="plot", **kwargs
):
    r"""This function takes a pandas `DataFrame` as input and plots the timeseries of each column.
    
    It first creates a plotly figure object. Then, it iterates over the columns of the `DataFrame` and adds a line plot for each column.
    The legend template of each line is customized to show the column name, the y-value, and the time for each point.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame containing timeseries simulation results which will be plotted.
    ylabel :
        `y`-axis label

    Other parameters
    ----------------
    plot_title : str, optional
        Title displayed on the top of the figure.
    filename : str, optional
        File name under which the plotly figure will be stored (in `png` format).
    folder : str, "plot_folder", optional
        Folder name where the plotly figure will be stored.
    **kwargs : dict, optional
        Every parameter found in save_fig function could also be added if needed.
    """
    fig = Figure()
    for col in data_df.columns:
        hovertemplate = _bold(col + '<br>' + ylabel + ': %{y:.2f}<br>Time: %{text}') + '<extra></extra>'
        fig.add_trace(go.Scatter(
            x=list(range(data_df.shape[0])), y=list(data_df[col]), mode='lines',
            name=_bold(col), hovertemplate=hovertemplate,
            text=_time_to_str(data_df.index)
        ))
    x_ticks = list(range(data_df.shape[0]))[::int(data_df.shape[0] / 6)]
    x_ticks_label = _time_to_str(data_df.index)[::int(data_df.shape[0] / 6)]
    save_fig(
        fig=fig, filename=filename, folder=folder, xlabel= "Time", ylabel=ylabel, plot_title=plot_title,
        x_ticks=(x_ticks,x_ticks_label), show_grid=True, **kwargs
    )


def _draw_traces(traces: list, showlegend: bool) -> Figure:
    r"""Intern function used to create a plotly figure from a list of trace created by pandasPower using `create_bus_trace`, `create_line_trace` and `create_trafo_trace`.

    Parameters
    ----------
    traces : list
        List of dicts which correspond to plotly traces.
    showlegend : bool
        If it is true, legend of traces will be displayed.

    Returns
    -------
    figure : graph_objs._figure.Figure
        The figure object is returned.
    """
    fig = Figure(data=traces,
                 layout=Layout(
                     showlegend=showlegend,
                     hovermode='closest',
                     margin=dict(b=5, l=5, r=5, t=5),
                     xaxis=XAxis(showgrid=False, zeroline=False, 
                                 showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, 
                                 showticklabels=False),
                     legend={'itemsizing': 'constant'},
                     ),
                )
    return fig


def save_fig(
        fig: Figure, filename: str = None, folder: str = "plot", plot_title: str = None,
        xlabel: str = None, ylabel: str = None, 
        x_ticks: tuple[list, list] = None,
        y_ticks: tuple[list, list] = None,
        width: int = 680, height: int = 400, title_x: float = 0.5, title_y: float = 0.97,
        legend_size: int = 12, tick_size: int = 12, axis_title_size: int = 12,
        title_size: int = 15, show_grid: bool = False, show_fig: bool = True):
    r"""Intern function used to format plotly layout, save and display figure.

    Parameters
    ----------
    fig : graph_objs._figure.Figure

    Other Parameters
    ----------------
    filename : str, optional
        File name under which the plotly figure will be stored (in `.png` format).
    folder : str, "plot"
        Folder name where the plotly figure will be stored.
    plot_title : str, optional
        Title displayed on the top of the figure.
    xlabel : str, optional
        `x`-axis label.
    ylabel : str, optional
        `y`-axis label.
    x_ticks : tuple[list, list], optional
        A pair of tick values and tick label for displaying `x`-axis.
    y_ticks : tuple[list, list], optional
        A pair of tick values and tick label for displaying `y`-axis.
    width : int, 680, optional
        Figure width (in pixels).
    height : int, 400, optional
        Figure height (in pixels).
    title_x : float, 0.5, optional
        Figure title `x`-axis location.
    title_y : float, 0.97, optional
        Figure title `y`-axis location.
    legend_size : int, 12, optional
        Legends characters size.
    tick_size : int, 12, optional
        Ticks characters size.
    axis_title_size : int, 12, optional
        Axis title characters size.
    title_size : int, 15, optional
        Figure title characters size.
    show_grid : bool, False, optional
        If it is `True`, add grid to the figure.
    show_fig : bool, True, optional
        If it is `True`, display figure in Jupyter notebook (usage with a `.py` file is not implemented).
    """
    fig.update_layout(
        font={"size": title_size}, xaxis_title=_bold(xlabel), yaxis_title=_bold(ylabel),
        xaxis=dict(tickfont=dict(size=tick_size, family="Arial Black"), zeroline=False),
        yaxis=dict(tickfont=dict(size=tick_size, family="Arial Black"), zeroline=False),
        paper_bgcolor='white', plot_bgcolor='white', width = width, height = height,
        xaxis_title_font = {"size": axis_title_size}, yaxis_title_font = {"size": axis_title_size},
        margin = dict(t=30), legend = {'font': {'size': legend_size}}
    )
    if x_ticks:
        fig.update_layout(
            xaxis=dict(tickmode = 'array', tickvals = x_ticks[0], ticktext = x_ticks[1])
        )
    if y_ticks:
        fig.update_layout(
            yaxis=dict(tickmode = 'array', tickvals = y_ticks[0], ticktext = y_ticks[1])
        )
    if show_grid:
        fig.update_layout(
            xaxis=dict(showline=True, showgrid=True, gridwidth= 1, linewidth=2, linecolor='black', gridcolor='black'),
            yaxis=dict(showline=True, showgrid=True, gridwidth= 1, linewidth=2, linecolor='black', gridcolor='black')
        )
    if plot_title:
        fig.update_layout(
            title={'text': _bold(plot_title), 'y': title_y, 'x': title_x, 'xanchor': 'center', 'yanchor': 'top'},
        )
    if filename:
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, filename)
        # plotly.offline.plot(fig, filename=file_path + ".html", auto_open=False)
        fig.write_image(file_path + ".png")
    if show_fig:
        fig.show()

def _bold(string: str | None) -> str | None:
    r"""Internal function to apply bold style to strings.

    Parameters
    ----------
    string : str or None
        Input string.

    Returns
    -------
    bold_string : str or None
        Input string with bold style applied.
    """
    if isinstance(string, str):
        return '<b>' + string + '<b>'
    else:
        return None

def _time_to_str(time_stamps: [time | datetime | list | tuple | pd.Index | pd.Series]) -> str | list[str] | None:
    r"""Internal function used to convert datetime objects in wanted string format (i.e. "12H00").

    Parameters
    ----------
    time_stamps : datetime.time | datetime.datetime | list | tuple | pandas.Index | pandas.Series
        Input datetime.

    Returns
    -------
    str_time_stamps : str or list[str] or None
        Datetime objects in wanted string format.

    """
    if isinstance(time_stamps, (time, datetime)):
        return time_stamps.strftime("%Hh%M")
    elif isinstance(time_stamps, (list, tuple, pd.Index, pd.Series)):
        return list(map(lambda x: x.strftime("%Hh%M"), time_stamps))
    else:
        return None