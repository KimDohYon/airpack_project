#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
Utility script to benchmark the data rate that a neural network will support.
"""

import os
import pathlib
from typing import Union

import numpy as np

import airpack.deploy.trt

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.deploy import trt, onnx
except ModuleNotFoundError as e:
    import sys
    _msg = f"{e}\nPlease run:\n  pip install -e {_airpack_root}"
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None


def benchmark(onnx_file: Union[str, os.PathLike]) -> None:
    """ Convert an onnx file to an optimized plan file and benchmark the maximum
    data rate throughput.

    .. seealso::
        This is a wrapper for :py:func:`airpack.deploy.trt.onnx2plan` and
        :py:func:`airpack.deploy.trt.plan_bench` that is setup with default values for
        the :py:func:`airpack.pytorch.model.Network`. If non-default parameters are
        needed for your customization, it is better practice to call directly the
        :py:func:`airpack.deploy.trt.onnx2plan` and
        :py:func:`airpack.pytorch.model.Network` instead of this demonstration
        wrapper.

    :param onnx_file:   Trained neural network saved as an onnx file
    :return:
    """
    onnx_file = pathlib.Path(onnx_file)  # convert from string if needed
    plan_file = onnx_file.with_suffix(".plan")

    # Make sure ONNX file exists
    if not onnx_file.exists() or onnx_file.is_dir():
        raise FileNotFoundError(f'ONNX model file not found: {onnx_file}')

    # If the PLAN file has not been created yet, do so
    if plan_file.exists():
        print(f"PLAN file already exists, reusing it: {plan_file}")
    else:
        print(f'Creating PLAN file: {plan_file}')
        trt.onnx2plan(
            onnx_file,
            input_node_name='input',
            input_port_name='',
            input_len=4096,
            fp16_mode=True,
            max_workspace_size=1073741824,
            max_batch_size=128,
            verbose=True
        )

    # Benchmark ONNX file
    print('\nBenchmarking ONNX file:')
    onnx.onnx_bench(
        onnx_file,
        cplx_samples=2048,
        batch_size=128,
        num_inferences=128,
        input_dtype=np.float32
    )

    # Benchmark PLAN file
    print('\nBenchmarking PLAN file:')
    airpack.deploy.trt.plan_bench(
        plan_file,
        cplx_samples=2048,
        batch_size=128,
        num_inferences=128,
        input_dtype=np.float32
    )


if __name__ == "__main__":
    # Default inference settings.
    _onnx_file = _airpack_root / "output" / "pytorch" / "saved_model.onnx"
    benchmark(_onnx_file)
