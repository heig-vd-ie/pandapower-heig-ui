
from datetime import time
from copy import deepcopy
import logging
import coloredlogs
import pandas as pd

import pandapower as pp
import pandapower.shortcircuit as sc

from src.pandapower_heig_ui import parse_grid_from_xlsx, plot_net_by_zone, load_power_profile, \
    create_output_writer, run_time_simulation, plot_timeseries_result, apply_power_profile,\
    plot_net_time_simulation_result, plot_net_short_circuit_result, _time_to_str

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


if __name__ == "__main__":
    net_file_path = r"./input_data/trey_data.xlsx"
    profile_file_path = r"./input_data/power_profile.xlsx"
    output_filename = "simulation_results"
    net: pp.pandapowerNet = parse_grid_from_xlsx(file_path=net_file_path)
    sc_net = deepcopy(net)
    sc_net["sgen"] = pd.DataFrame(columns=sc_net["sgen"].columns)
    plot_net_by_zone(net=net, plot_title="Trey grid by zone", filename="grid_by_zone", folder=".cache", show_fig=False)
    time_series: dict = load_power_profile(file_path=profile_file_path)

    for eq in ["load", "sgen"]:
        apply_power_profile(net=net, equipment=eq, power_profiles=time_series[eq])
    create_output_writer(net=net)
    result_df = run_time_simulation(net=net, folder=".cache", output_filename=output_filename)

    plot_timeseries_result(data_df=result_df["res_bus.vm_pu"], ylabel="V [pu]", folder=".cache",
                           plot_title="Bus voltage", filename= "voltage_result", show_fig=False)
    plot_timeseries_result(data_df=result_df["res_line.loading_percent"], ylabel="[%]", plot_title="Line loading",
                           filename= "line_loading_result", folder=".cache", legend_size=20, show_fig=False)

    for plot_time in [time(hour=4), time(hour=20)]:
        plot_net_time_simulation_result(net=net, filename="sim_result_" + _time_to_str(plot_time) ,
                                        plot_time=plot_time, folder=".cache", show_fig=False)

    for fault in ["3ph", "2ph", "1ph"]:
        sc.calc_sc(net,topology="radial", fault=fault, ip=True)
        plot_net_short_circuit_result(
            net=net, filename=fault + "_sc_results", plot_title=fault +" short-circuit results", folder=".cache",
            show_fig = False
        )

