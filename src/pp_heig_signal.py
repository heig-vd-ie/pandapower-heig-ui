import pandas as pd
from scipy.fft import fft
import math
import numpy as np
import logging
import coloredlogs

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

def compute_fft(data: pd.DataFrame, time_column: str, f: float = 50, round_time: int = 9, round_freq: int = 4) \
    -> pd.DataFrame:
    r"""Compute fast fourier transform of signals with timeseries. 

    Parameters
    ----------
    data : pandas.DataFrame
        Table where signal data with timeseries in seconds is recorded
    time_column : str
        Column name where timeseries is stored
    
    Other Parameters
    ----------------
    f : float, 50, optional
        Fundamental frequency
    round_time: int, 9, optional
        Digit maximum precision of the time vector (Set to ns avoids numerical issues)
    round_freq: int, 4, optional 
        Digit maximum precision of the frequency vector (Set to 100 us avoids numerical issues)

    Returns
    -------
    fft_df : pandas.DataFrame
        Table containing the fast fourier transform modules and the corresponding frequency in Hz.

    Example
    -------
    Loading data recoded from oscilloscope
    
    >>> data = pd.read_csv("data.csv", skiprows=12, names=['time', 'U_in', 'U_out', 'I_in', 'I_out'])
    >>> data_fft =  compute_phasor(data=data, time_column="time")
    """
    results: pd.DataFrame = pd.DataFrame()
    if data[time_column].diff().dropna().round(round_time).unique().shape[0] != 1 : 
        raise RuntimeError("Time steps are not homogenous")
    h: float = data[time_column].iat[1] - data[time_column].iat[0]
    f_sample: float = 1/h
    nb_period: int = math.floor((data[time_column].iat[-1] - data[time_column].iat[0]) *f)
    if nb_period == 0:
        raise RuntimeError("Signal contains less than one period")
    signal_len: int = int(f_sample/f*nb_period)
    results["freq"] = (f_sample/signal_len*np.arange(int(signal_len/2))).round(round_freq)
    for col in data.columns.difference([time_column]):
        yf = fft(data[col].iloc[0: signal_len].values)
        results[col] = 2.0/signal_len*np.abs(yf[0:int(signal_len/2)]) # type: ignore
    return results


def compute_phasor(data: pd.DataFrame, time_column: str, f: float = 50, round_time: int = 9) -> dict: 
    r"""Compute fundamental component complex phasors of signals with timeseries. 

    Parameters
    ----------
    data : pandas.DataFrame
        Table where signal data with timeseries in seconds is recorded
    time_column : str
        Column name where timeseries is stored
    
    Other Parameters
    ----------------
    f : float, 50, optional
        Fundamental frequency
    round_time: int, 9, optional
        Digit maximum precision of the time vector (Set to ns avoids numerical issues)

    Returns
    -------
    phasors : dict
        Dictionary containing fundamental component complex phasors
    Example
    -------
    Loading data recoded from oscilloscope

    >>> data = pd.read_csv("data.csv", skiprows=12, names=['time', 'U_in', 'U_out', 'I_in', 'I_out'])
    >>> data_fft =  compute_phasor(data=data, time_column="time")
    """
    results: dict = {}
    if data[time_column].diff().dropna().round(round_time).unique().shape[0] != 1 : 
        raise RuntimeError("Time steps are not homogenous")
    h: float = data[time_column].iat[1] - data[time_column].iat[0]
    f_sample: float = 1/h
    nb_period: int = math.floor((data[time_column].iat[-1] - data[time_column].iat[0]) *f)
    if nb_period == 0:
        raise RuntimeError("Signal contains less than one period")
    signal_len: int = int(f_sample/f*nb_period)
    start_angle: float = 0
    f_idx: int = int(f*signal_len/f_sample)
    for i, col in enumerate(data.columns.difference([time_column])):
        signal_fft: np.complex = 2/signal_len*fft(data[col].iloc[0: signal_len].values)[f_idx]
        if i == 0:
            start_angle = np.angle(signal_fft)
        results[col] = signal_fft/complex(np.cos(start_angle), np.sin(start_angle))
    return results