#!/usr/bin/env python3
#
# Copyright 2021, Deepwave Digital, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
This application is used as an example on how to deploy a neural network for inference
on the AIR-T. The method provided here leverages the PyCUDA interface for the shared
memory buffer. PyCUDA is installed by default in AirStack.
"""

import os
import sys
import pathlib
from typing import Union
import numpy as np
import argparse
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.deploy import trt, trt_utils
except ModuleNotFoundError as e:
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None


def _parse_command_line_arguments():
    """ Parses command line input arguments for airt_infer

    :return:
    """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='AIR-T Inference function.',
                                     formatter_class=help_formatter)
    parser.add_argument('-m', type=str, required=True, dest='model_file',
                        help='Trained neural network model file (onnx, uff, or plan)')
    parser.add_argument('-l', type=int, required=False, dest='cplx_input_len',
                        default=2048,
                        help='Complex samples per infer, half the input_len of model')
    parser.add_argument('-b', type=int, required=False, dest='batch_size',
                        default=128, help='Mini-batch size for inference')
    parser.add_argument('-n', type=int, required=False, dest='num_batches',
                        default=-1,
                        help='Number of batches to execute. Use -1 for continuous')
    parser.add_argument('-s', type=float, required=False, dest='samp_rate',
                        default=31.25e6, help='Radio sample rate')
    parser.add_argument('-f', type=float, required=False, dest='freq',
                        default=2400e6, help='Tuning frequency of radio in Hz')
    parser.add_argument('-c', type=int, required=False, dest='chan',
                        default=0, help='Radio channel to receive on (0 or 1)')
    parser.add_argument('-i', type=str, required=False, dest='input_node_name',
                        default='input', help='Name of input node')
    parser.add_argument('-p', type=str, required=False, dest='input_port_name',
                        default='', help='Name of input port')

    return parser.parse_args(sys.argv[1:])


def airt_infer(model_file: Union[str, os.PathLike], cplx_input_len: int = 2048,
               batch_size: int = 128, num_batches: int = -1, samp_rate: float = 31.25e6,
               freq: float = 2400e6, chan: int = 0, input_node_name: str = 'input',
               input_port_name: str = '',) -> None:
    """ Function to receive samples and perform inference using the AIR-T

    The function will input a model file (onnx, uff, or plan), optimize it if necessary
    using TensorRT, setup the shared memory infrastructure between the radio buffer and
    the cudnn/TensorRT inference, configure the AIR-T's radio hardware, stream samples
    from the radio to the inference engine, and print the inference result.

    .. seealso ::

        The :py:func:`airt_infer` function requires a trained model to perform inference.
        This may be obtained by running the :py:func:`airpack_scripts.tf2.run_training.train`
        function to get an onnx file.

    Example usage for :py:func:`airpack.tf2.model.default_network`:

    .. code-block:: python

        from airpack_scripts.airt.run_airt_inference import airt_infer
        onnx_file = saved_model.onnx  # Trained model file
        airt_infer(
            onnx_file,
            cplx_input_len=2048,
            batch_size=128,
            num_batches=-1,
            samp_rate=31.25e6,
            freq=2.4e9,
            chan=0,
            input_node_name='input',
            input_port_name=''
        )

    .. note::
        This function expects the input to be defined in number of complex samples even
        though the DNN expects real samples. The number of real samples is calculated to
        be ``real_input_len = 2 * cplx_input_len``.

    .. note::
        The AIR-T's radio  will produce data of type SOAPY_SDR_CF32 which is the same as
        np.complex64. Because a np.complex64 value has the same memory construct as two
        np.float32 values, the GPU memory buffer is defined as twice the size of the SDR
        buffer but np.float32 dtypes. This is done because the neural network expects an
        input of np.float32. The SOAPY_SDR_CF32 can be copied directly to the np.float32
        buffer.

    .. note::
        This utility uses PyCUDA to create a shared memory buffer (zero-copy) that will
        receive samples from the AIR-T's radio to be fed into the DNN. Note that this
        buffer is shared between the SDR and the DNN to prevent unnecessary copies. The
        buffer fed into the DNN for inference will be a 1-dimensional array that contains
        the samples for an entire mini-batch with length defined below. For example an
        ``input_len = 2048`` complex samples and a ``batch_size = 128`` will have a buffer
        of size ``2048 * 128 = 262,144`` complex samples.

    :param model_file:      Trained neural network model file (onnx, uff, or plan)
    :param cplx_input_len:  Complex samples per inference
    :param batch_size:      Mini-batch size for inference
    :param num_batches:     Number of batches to execute. -1 -> Infinite
    :param samp_rate:       Radio's sample rate in samples per second
    :param freq:            Tuning frequency of radio in Hz
    :param chan:            Radio channel to receive on (0 or 1)
    :param input_node_name: Name of input node
    :param input_port_name: Name of input node port (for TensorFlow 1)
    :return:                None
    """

    model_file = pathlib.Path(model_file)  # convert from string if needed
    real_input_len = 2 * cplx_input_len

    # Optimize model if given an onnx or uff file
    if model_file.suffix == '.plan':
        plan_file_name = model_file
    elif model_file.suffix == '.onnx':
        plan_file_name = trt.onnx2plan(model_file,
                                       input_node_name=input_node_name,
                                       input_port_name=input_port_name,
                                       input_len=real_input_len,
                                       max_batch_size=batch_size)
    elif model_file.suffix == '.uff':
        plan_file_name = trt.uff2plan(model_file,
                                      input_len=real_input_len,
                                      max_batch_size=batch_size)
    else:
        raise ValueError(f'Unknown file extension {model_file.suffix}')

    # Setup the CUDA context
    trt_utils.make_cuda_context()

    # Setup the shared memory buffer
    buff_len = 2 * cplx_input_len * batch_size
    sample_buffer = trt_utils.MappedBuffer(buff_len, np.float32)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = trt.TrtInferFromPlan(plan_file_name, batch_size, sample_buffer)

    # Create, configure, and activate AIR-T's radio hardware
    sdr = Device()
    sdr.setGainMode(SOAPY_SDR_RX, chan, True)
    sdr.setSampleRate(SOAPY_SDR_RX, chan, samp_rate)
    sdr.setFrequency(SOAPY_SDR_RX, chan, freq)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])
    sdr.activateStream(rx_stream)

    # Start receiving signals and performing inference
    print('Receiving Data')
    ctr = 0
    while ctr < num_batches or num_batches == -1:
        try:
            # Receive samples from the AIR-T buffer
            sr = sdr.readStream(rx_stream, [sample_buffer.host], buff_len)
            if sr.ret == SOAPY_SDR_OVERFLOW:  # Data was dropped, i.e., overflow
                print('O', end='', flush=True)
                continue
            # Run samples through neural network
            dnn.feed_forward()
            # Get data from DNN output layer.
            output_arr = dnn.output_buff.host
            # Reshape into matrix of shape = (batch_size, ouput_len)
            output_mat = output_arr.reshape(batch_size, -1)
            # Determine what the predicted signal class is for each window
            infer_result = np.argmax(output_mat, axis=1)
            # Determine the unique class values found in current batch
            classes_found = np.unique(infer_result)
            print(f'Signal classes found for batch {ctr} = {classes_found}')
        except KeyboardInterrupt:
            break
        ctr += 1
    sdr.closeStream(rx_stream)


if __name__ == '__main__':
    _pars = _parse_command_line_arguments()
    airt_infer(
        _pars.model_file,
        cplx_input_len=_pars.cplx_input_len,
        batch_size=_pars.batch_size,
        num_batches=_pars.num_batches,
        samp_rate=_pars.samp_rate,
        freq=_pars.freq,
        chan=_pars.chan,
        input_node_name=_pars.input_node_name,
        input_port_name=_pars.input_port_name
    )
