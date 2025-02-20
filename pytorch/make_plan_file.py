#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import Union

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.deploy import trt
except ModuleNotFoundError as e:
    import sys
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None


def convert(onnx_file: Union[str, os.PathLike]) -> os.PathLike:
    """ Converts an onnx file to an optimized plan file using TensorRT. This is a
    wrapper for :py:func:`airpack.deploy.trt.onnx2plan` that is setup with default
    values for the :py:class:`airpack.pytorch.model.Network`.

    .. seealso::
        If non-default parameters are needed for your customization, it is better
        practice to call :py:func:`airpack.deploy.trt.onnx2plan` instead of this
        demonstration wrapper for the :py:class:`airpack.pytorch.model.Network`.

    :param onnx_file:   Trained neural network saved as an onnx file
    :return:            Name of saved .plan file
    """
    plan_file = trt.onnx2plan(
        onnx_file,
        input_node_name='input',
        input_port_name='',
        input_len=4096,
        fp16_mode=True,
        max_workspace_size=1073741824,
        max_batch_size=128,
        verbose=True
    )
    return plan_file


if __name__ == '__main__':
    _onnx_file = _airpack_root / "output" / "pytorch" / "saved_model.onnx"
    convert(_onnx_file)
