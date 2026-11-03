import json
import os
import sys

import numpy as np
import pytest

from agamemnon import tcn

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import train_tcn  # noqa: E402


def test_numpy_inference_matches_torch(tmp_path):
    torch.manual_seed(1)
    net = train_tcn.Net(tcn.N_INPUTS).double().eval()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(700, tcn.N_INPUTS))
    mean, std = rng.normal(size=tcn.N_INPUTS), rng.uniform(0.5, 2, tcn.N_INPUTS)
    path = tmp_path / "w.json"
    path.write_text(json.dumps(train_tcn.export(net, mean, std)))
    loaded = tcn.load(str(path))
    ours = tcn.forward(loaded, x)
    with torch.no_grad():
        xt = torch.tensor(((x - mean) / std).T[None])
        theirs = net(xt)[0].numpy().T
    assert np.allclose(ours, theirs, atol=1e-8)


def test_committed_weights_load_if_present():
    if not os.path.isfile(tcn.WEIGHTS):
        pytest.skip("no trained weights yet")
    net = tcn.load()
    assert len(net["mean"]) == tcn.N_INPUTS
