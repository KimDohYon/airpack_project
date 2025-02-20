#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import Union
from tqdm import tqdm
import torch

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.pytorch import fileio, model
except ModuleNotFoundError as e:
    import sys
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None

DEVICE = torch.device("cpu")

def train(network: torch.nn.Module, dataloader: torch.utils.data.dataloader,
          optimizer: torch.optim, criterion: torch.nn, scalar: int) -> tuple:
    """ Train pytorch neural network and return loss and accuracy

    :param network:     torch model
    :param dataloader:  torch data loader
    :param optimizer:   torch optimizer function
    :param criterion:   torch loss function
    :param scalar:      scalar integer to normalize data
    :return:            ``(train_loss, train_acc)``
    """
    network.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for data_in, labels in dataloader:
        data_in = data_in.to(DEVICE) # copy to GPU
        labels = labels.to(DEVICE)
        data_fix = torch.view_as_real(data_in * scalar)
        pred = network(data_fix)
        loss = criterion(pred, labels)
        running_loss += loss.item()
        _, pred_label = torch.max(pred, 1)
        running_accuracy += torch.sum(pred_label == labels)
        optimizer.zero_grad()  # Zero out gradients
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    train_acc = running_accuracy/len(dataloader.dataset)
    return train_loss, train_acc

def validate(network: torch.nn.Module, dataloader: torch.utils.data.dataloader,
             criterion: torch.nn, scalar: int) -> tuple:
    """ Train pytorch neural network and return loss and accuracy

    :param network:     torch model
    :param dataloader:  torch data loader
    :param criterion:   torch loss function
    :param:             scalar int to normalize data
    :return:            ``(val_loss, val_acc)``
    """
    network.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for data_in, labels in dataloader:
            data_in = data_in.to(DEVICE)
            labels = labels.to(DEVICE)
            data_fix = torch.view_as_real(data_in *scalar)
            pred = network(data_fix)
            loss = criterion(pred, labels)
            running_loss += loss.item()
            _, pred_label = torch.max(pred, 1)
            running_accuracy += torch.sum(pred_label == labels)
    val_loss = running_loss/len(dataloader.dataset)
    val_acc = running_accuracy/len(dataloader.dataset)
    return val_loss, val_acc

def main(data_folder: Union[str, os.PathLike], n_epoch: int = 10) -> float:
    """ Script used to train :py:class:`airpack.pytorch.model.Network`.

    .. note ::
        You may modify the parameters in this script to tune the hyperparameters of the
        :py:func:`airpack.pytorch.model.Network`.

    :param data_folder:     Location of training data
    :param n_epoch:         Number of epochs in training process
    :return:                Training accuracy
    """
    data_folder = pathlib.Path(data_folder)
    train_data_folder = data_folder / "train"   # Location of training data
    test_data_folder = data_folder / "test"    # Location of training data (validation subset)
    model_save_folder = _airpack_root / "output" / "pytorch"
    os.makedirs(model_save_folder, exist_ok=True)
    input_len = 2048  # signal Window length in complex samples
    output_len = 12  # Number of signal classes
    learning_rate = 1e-3    # Learning rate
    batch_size = 64  # Number of signals to average for each training calculation
    normalize_scalar = 1000 # Scalar multiplied by input to normalize the data

    # Import Data Sets
    train_data = fileio.load_waveform(train_data_folder, input_len, True)
    test_data = fileio.load_waveform(test_data_folder, input_len, False)

    # Import Loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Define Feed forward CNN model
    classifier = model.default_network(input_len, output_len).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Start Training Process
    for epoch in tqdm(range(n_epoch), total=n_epoch, desc='Training Progress', unit='epoch'):
        print('Epoch {} of {}'.format(epoch+1, n_epoch))
        train_epoch_loss, train_epoch_accr = train(classifier, train_loader, optimizer, criterion, normalize_scalar)
        val_epoch_loss, val_epoch_accr = validate(classifier, test_loader, criterion, normalize_scalar)
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'
              .format(train_epoch_loss, train_epoch_accr))
        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'
              .format(val_epoch_loss, val_epoch_accr))

    # Save model as onnx
    dummy_input = classifier.get_dummy_data(batch_size).to(DEVICE)
    output_file = os.path.join(model_save_folder, 'saved_model.onnx')
    # Create ONNX file
    torch.onnx.export(classifier, dummy_input, output_file, export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: 'batch_size'},
                                    "output": {0: 'batch_size'}})
    return train_epoch_accr

if __name__ == '__main__':
    _default_data_folder = '/data'
    main(_default_data_folder)
