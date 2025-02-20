#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import pkg_resources
import torch

_parse_version = pkg_resources.packaging.version.parse
_torch_version = _parse_version(torch.__version__)

if _torch_version < _parse_version('1.7.0'):
    raise ImportError(
        "airpack.pytorch requires at least PyTorch 1.7.0, found version {}".format(
            torch.__version__))