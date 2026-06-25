"""Compatibility shim: basicsr 1.4.2 (a transitive dependency of GFPGAN and
Real-ESRGAN) imports `torchvision.transforms.functional_tensor`, which was
removed in torchvision >= 0.17. There is no newer basicsr release that fixes
this for the torch/torchvision versions Python 3.12 requires, so we register a
drop-in module under that name backed by the current `functional` module before
anything imports basicsr. Must be imported before `basicsr`/`gfpgan`/`realesrgan`.
"""
from __future__ import annotations

import sys
import types

if "torchvision.transforms.functional_tensor" not in sys.modules:
    from torchvision.transforms import functional as _F

    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
