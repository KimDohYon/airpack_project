#! /usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

class WaveformDataset(torch.utils.data.Dataset):

    _default_device: torch.device = torch.device('cpu')

    def __init__(self,
                 root: Union[Path, str],
                 dtype: Any,
                 num_samples: Optional[int] = None,
                 device: torch.device = torch.device('cpu'),
                 transform: Optional[torch.nn.Module] = None,
                 seed: Union[None, int, np.random.SeedSequence] = None,
                 shuffle: bool = False) -> None:
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype
        self.root = Path(root).expanduser()
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.file_list = sorted(self.root.rglob('*.bin'), key=lambda p: p.stem)

        if len(self.file_list) == 0:
            raise FileNotFoundError(
                "No data files found in {}".format(root))

        if shuffle:
            self.rng.shuffle(self.file_list)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, Dict]:
        """Load the n-th item from the dataset.

        .. note ::
            The shape of the returned waveform tensor will be whatever
            is returned by the provided transform. If the transform is None,
            it will be the shape returned by ``_load_item()``.

        :param n: The index of the item (file) to be loaded
        :return: tuple: ``(waveform, labels)``
        """
        path = self.file_list[n].resolve()
        waveform, labels = self._load_item(path.stem, path.suffix, path.parent)
        if self.num_samples and waveform.shape[-1] != self.num_samples:
            error_fmt = "Invalid number of samples in file (required {}, found {}): {}"
            raise ValueError(error_fmt.format(self.num_samples, waveform.shape[-1], path))
        if self.transform is not None:
            with torch.no_grad():
                waveform = self.transform(waveform)
        return waveform, labels

    def __len__(self) -> int:
        """Number of samples in this dataset.

        .. note ::
            For a WaveformFolderDataset, the number of samples is equal to the
            number of data files contained in the folder.
        """
        return len(self.file_list)

    def _load_item(self, fileid: str, suffix: str,
                   path: Path) -> Tuple[torch.Tensor, Dict]:
        """Actually does the work of loading a file.

        .. note ::
            The numeric label is the name of the bottom-most folder in the tree of data:
            e.g., for file 'data/train/x/y/file.bin', the label is 'y'.
            Regardless of the type of the data being loaded, the tensor returned by
            this method will be ``dtype=torch.complex64``.

        :param fileid:  Filename (without extension)
        :param suffix:  Filename extension
        :param path:    Directory that this item resides in
        :return: tuple: ``(waveform, label)``
        """

        #labels = dict(field.split("=", 2) for field in fileid.split("_")) # this needs to just be a number
        filename = os.path.join(path, fileid + suffix)
        label = int(os.path.basename(os.path.dirname(filename)))
        waveform = np.fromfile(filename, dtype=self.dtype)

        if self.dtype == np.complex128:
            # This technically loses precision, but we never really want to
            # do double-precision DSP nor use it as input to our ML models.
            waveform = np.asarray(waveform, dtype=np.complex64)
        elif self.dtype != np.complex64:
            waveform = np.asarray(waveform[0::2] + 1j * waveform[1::2],
                                  dtype=np.complex64)

        # Sample scaling to [-1, 1]
        if self.dtype in (np.int8, np.int16, np.int32):
            # scale factor for the data type. for example, for int16,
            # range is [-32768, 32767], so 1/(max+1) = 1/32768
            scale_factor = 1 / (np.iinfo(self.dtype).max + 1)
            waveform = waveform * scale_factor

        # Create the tensor and add the channel dimension. This dataset
        # contains single-channel data.
        _num_channels = 1
        _samples_per_channel = int(len(waveform) / _num_channels)
        _tensor_shape = (_num_channels, _samples_per_channel)
        tensor = torch.as_tensor(waveform, device=torch.device('cpu')).view(_tensor_shape)

        return tensor, label

def load_waveform(data_folder: str, num_samples: float, shuffle: bool) \
 -> torch.utils.data.Dataset:
    """ Read in signal data, transform it, and return pytorch dataset

    :param data_folder:  Directory of signal data
    :param num_samples:  Signal Window length in complex samples
    :param shuffle:      Boolean to shuffle samples
    :return:             data

    """
    data = WaveformDataset(root=data_folder,
                                   dtype=np.int16,
                                   shuffle=shuffle,
                                   num_samples=num_samples)
    return data

if __name__ == "__main__":
    load_waveform("/data", 2048, True)