import pandas as pd
from scipy.fft import fft
import math
import numpy as np
import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

def calcul_fft(data: pd.DataFrame, time_column: str, f: float = 50, round_time: int = 9, round_freq: int = 4) \
    -> pd.DataFrame:
    results: pd.DataFrame = pd.DataFrame()
    if data[time_column].diff().dropna().round(round_time).unique().shape[0] != 1 : 
        log.error("Time steps are not homogenous")
    h: float = data[time_column].iat[1] - data[time_column].iat[0]
    f_sample: float = 1/h
    nb_period: int = math.floor((data[time_column].iat[-1] - data[time_column].iat[0]) *f)
    signal_len: int = int(f_sample/f*nb_period)
    results["freq"] = (f_sample/signal_len*np.arange(int(signal_len/2))).round(round_freq)
    for col in data.columns.difference([time_column]):
        yf = fft(data[col].iloc[0: signal_len].values)
        results[col] = 2.0/signal_len*np.abs(yf[0:int(signal_len/2)]) # type: ignore
    return results


def calcul_phasor(data: pd.DataFrame, time_column: str, f: float = 50, round_phasor: int = 4) -> dict: 

    results: dict = {}
    h: float = data[time_column].iat[1] - data[time_column].iat[0]
    f_sample: float = 1/h
    nb_period: int = math.floor((data[time_column].iat[-1] - data[time_column].iat[0]) *f)
    signal_len: int = int(f_sample/f*nb_period)
    start_angle: float = 0
    f_idx: int = int(f*signal_len/f_sample)
    for i, col in enumerate(data.columns.difference([time_column])):
        signal_fft: np.complex = 2/signal_len*fft(data[col].iloc[0: signal_len].values)[f_idx]
        if i == 0:
            start_angle = np.angle(signal_fft)
        signal_fft = signal_fft/complex(np.cos(start_angle), np.sin(start_angle))
        results[col] = {
            "module": round(np.abs(signal_fft), round_phasor), "angle": round(np.angle(signal_fft), round_phasor)
            }
    return results