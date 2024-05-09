from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

import os


class SummaCA(BaseDataset):
    """Data set class for the SUMMA CA data set by [#]_ and [#]_.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
        
    References
    ----------
    .. [#] 
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(SummaCA, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        
        # print("Loading basin data for basin", basin)
        
        # get forcings
        df = load_summa_ca_forcings(self.cfg.data_dir, basin)

        return df

def load_summa_ca_forcings(data_dir: Path, basin: str) -> Tuple[pd.DataFrame, int]:
    """Load the forcing data for a basin of the SUMMA CA data set.

    Parameters
    ----------
    data_dir : Path
        Path to the SUMMA CA directory. This folder must contain a 'summa_input-output_LSTM_1D' folder containing the forcing files (.csv), starting with the 8-digit basin 
        id or id name (ex: "71028585_" or "bowAtBanff_").
    basin : str
        8-digit USGS identifier of the basin or string name of the basin.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    int
        Catchment area (m2), specified in the header of the forcing file.
    """
    
    forcing_path = data_dir / 'summa_input-output_LSTM_1D'
    
    try:
        file_path = forcing_path / f'{basin}_forcing_inp-out.csv'
    except:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    with open(file_path, 'r') as fp:
        # load the dataframe from the rest of the stream
        df = pd.read_csv(fp)
        
        # Normalize date to midnight
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        
        df = df.set_index("date")
        
    return df