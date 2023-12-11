
from datetime import time
from copy import deepcopy
import logging
import coloredlogs
import pandas as pd

import pandapower as pp
import pandapower.shortcircuit as sc

# from pp_heig_plot import (
#     plot_timeseries_result, plot_short_circuit_result, plot_power_network,
#     plot_timestep_powerflow_result, _time_to_str
# )
from pp_heig_simulation import load_net_from_xlsx, load_power_profile_form_xlsx
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


if __name__ == "__main__":
    net_file_path = r"./tests/input_data/trey_data.xlsx"
    profile_file_path = r"./tests/input_data/power_profile.xlsx"
    output_filename = "simulation_results"
    net: pp.pandapowerNet = load_net_from_xlsx(file_path=net_file_path)
    # sc_net = deepcopy(net)
    # sc_net["sgen"] = pd.DataFrame(columns=sc_net["sgen"].columns)
    # plot_power_network(
    #     net=net, plot_title="Trey grid by zone", filename="grid_by_zone", folder=".cache", show_fig=False
    # )
    time_series: dict = load_power_profile_form_xlsx(file_path=profile_file_path)

    # for eq in ["load", "sgen"]:
    #     apply_power_profile(net=net, equipment=eq, power_profiles=time_series[eq])
    # create_output_writer(net=net)
    # result_df = run_time_simulation(net=net, folder=".cache", output_filename=output_filename)

    # plot_timeseries_result(data_df=result_df["res_bus.vm_pu"], ylabel="V [pu]", folder=".cache",
    #                        plot_title="Bus voltage", filename= "voltage_result", show_fig=False)
    # plot_timeseries_result(data_df=result_df["res_line.loading_percent"], ylabel="[%]", plot_title="Line loading",
    #                        filename= "line_loading_result", folder=".cache", legend_size=20, show_fig=False)

    # for plot_time in [time(hour=4), time(hour=20)]:
    #     plot_timestep_powerflow_result(net=net, filename="sim_result_" + _time_to_str(plot_time),
    #                                     plot_time=plot_time, folder=".cache", show_fig=False)

    # for fault in ["3ph", "2ph", "1ph"]:
    #     sc.calc_sc(net,topology="radial", fault=fault, ip=True)
    #     plot_short_circuit_result(
    #         net=net, filename=fault + "_sc_results", plot_title=fault +" short-circuit results", folder=".cache",
    #         show_fig=False
    #     )

