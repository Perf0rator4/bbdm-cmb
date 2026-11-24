"""Tiny CPU checks of the joint training path, gradients, freezing and export."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from bbdm.data import crop_audit, forbidden_rects, tile_patch_positions
from bbdm.data.joint import JointPatchDataset
from bbdm.model import BBDM, UNet
from bbdm.model.detection_loss import CFAHeatmapLoss, preprocess_cfa
from bbdm.sample import run_inference, sample_batch
from bbdm.train import load_checkpoint, train
from cfa_unet_inference import CFAUNetInference
from test_make_test_set import _write_tiles
from test_region import P, BOX, _names, _sky


class TinyCFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


@pytest.fixture
def objective(tmp_path, monkeypatch):
    import bbdm.model.detection_loss as module
    torch.manual_seed(3)
    monkeypatch.setattr(module, "CrossFrequencyAttentionUNet", TinyCFA)
    path = tmp_path / "cfa.pt"
    torch.save(TinyCFA().state_dict(), path)
    loss = CFAHeatmapLoss(path, [10, 20, 30], [50, 60, 70], edge_margin=1)
    return loss, path


def model():
    return BBDM(UNet(in_ch=3, cond_ch=3, base_ch=8, time_dim=32, groups=4,
                     upsample="resize_conv"), T=10)


def test_torch_preprocessing_matches_cfa_and_keeps_pixel_gradients():
    images = np.random.default_rng(7).normal(10, 80, (2, 3, 16, 16)).astype(np.float32)
    images[0, 1, 0, 0] = 10000
    images[1, 2] = 40
    tensor = torch.tensor(images, requires_grad=True)
    expected = np.stack([CFAUNetInference._preprocess_tile(image) for image in images])
    processed = preprocess_cfa(tensor)
    np.testing.assert_allclose(processed.detach().numpy(), expected, atol=2e-7)
    processed.sum().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_loss_changes_diffusion_gradient_but_not_detector(objective):
    loss, _ = objective
    bridge = model()
    generator = torch.Generator().manual_seed(5)
    x, y = torch.randn(2, 3, 16, 16, generator=generator), torch.randn(2, 3, 16, 16, generator=generator)
    before = {k: v.clone() for k, v in loss.detector.state_dict().items()}
    loss.train()
    assert not loss.detector.training
    base = bridge.loss(x, y, generator=torch.Generator().manual_seed(4))
    base.backward()
    baseline_grad = bridge.model.out_conv.weight.grad.clone()
    bridge.zero_grad()
    total, terms = bridge.loss(x, y, detection_loss=loss, detection_weight=1000,
                               detection_steps=2, return_terms=True,
                               generator=torch.Generator().manual_seed(4))
    total.backward()
    assert terms["detection"] > 0 and torch.isfinite(total)
    assert not torch.allclose(bridge.model.out_conv.weight.grad, baseline_grad, atol=1e-9, rtol=1e-6)
    assert all(p.grad is None and not p.requires_grad for p in loss.detector.parameters())
    assert all(torch.equal(before[k], v) for k, v in loss.detector.state_dict().items())
    bridge.zero_grad()
    zero = bridge.loss(x, y, detection_loss=loss, detection_weight=0,
                       generator=torch.Generator().manual_seed(4))
    assert torch.equal(zero, base.detach())


def test_gradient_sampler_is_identical_to_inference_sampler():
    bridge = model()
    planck = torch.randn(1, 3, 16, 16)
    expected = bridge.sample(planck, S=3, generator=torch.Generator().manual_seed(11))
    actual = bridge.sample_with_grad(planck, S=3, generator=torch.Generator().manual_seed(11))
    assert torch.equal(expected, actual) and actual.requires_grad
    actual.square().mean().backward()
    assert bridge.model.out_conv.weight.grad.abs().sum() > 0


def test_joint_crops_share_pixels_augmentation_and_filters(tmp_path):
    _write_tiles(str(tmp_path), _sky(7))
    positions = {pos for name in _names() for pos in tile_patch_positions(name, P)}
    ds = JointPatchDataset(str(tmp_path), [1, 20, -30], [2, 4, 6], positions=positions,
                           patch_size=P, augment=True)
    x, y = ds[1]
    assert x.shape == y.shape == (3, P, P)
    physical = ds.denormalize(x.numpy())
    np.testing.assert_allclose(physical[0], physical[2], atol=2e-5)
    forbidden = [(0, P, 0, P)]
    crops = JointPatchDataset(str(tmp_path), [1, 20, -30], [2, 4, 6], forbidden=forbidden,
                              n_samples=3, patch_size=P, augment=True)
    for i in range(3):
        x, _ = crops[i]
        physical = crops.denormalize(x.numpy())
        np.testing.assert_allclose(physical[0], physical[2], atol=2e-5)
    audit = crop_audit(crops, {"val": {(0, 0)}, "test": set()}, n=100)
    assert audit["crops_touching_val"] == 0
    # One missing frequency cannot be hidden by averaging its mask fraction over channels.
    for path in (tmp_path / "Planck+ACT" / "f220").glob("*.fits"):
        with fits.open(path, mode="update") as hdus:
            hdus[0].data[:] = 0
    with pytest.raises(ValueError, match="No common valid"):
        JointPatchDataset(str(tmp_path), [0] * 3, [1] * 3, positions=positions, patch_size=P)


def test_joint_grid_mismatch_is_rejected(tmp_path):
    _write_tiles(str(tmp_path), _sky(7))
    path = next((tmp_path / "Planck" / "f143").glob("*.fits"))
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["CRVAL1"] = 20
    with pytest.raises(ValueError, match="WCS"):
        JointPatchDataset(str(tmp_path), [0] * 3, [1] * 3, n_samples=1, patch_size=P)


class SmallDataset:
    mu, sigma = np.array([10, 20, 30]), np.array([50, 60, 70])

    def __init__(self):
        g = torch.Generator().manual_seed(6)
        self.x, self.y = torch.randn(2, 3, 16, 16, generator=g), torch.randn(2, 3, 16, 16, generator=g)

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def test_training_checkpoint_monitor_resume_and_joint_inference(tmp_path, objective):
    loss, _ = objective
    ds = SmallDataset()
    kwargs = dict(n_epochs=1, batch_size=2, ema_start=0, num_workers=0, device="cpu",
                  checkpoint_dir=str(tmp_path / "run"), detection_loss=loss, detection_weight=1,
                  detection_steps=2, monitor_dataset=ds, monitor_n=2, monitor_S=2, monitor_every=1)
    train(model(), ds, ds, **kwargs)
    loaded, ckpt = load_checkpoint(tmp_path / "run" / "last.pt", device="cpu")
    assert ckpt["train_detection"] > 0 and ckpt["val_detection"] > 0
    assert len(ckpt["monitor"][0]["per_channel"]) == 3
    assert ckpt["monitor"][0]["detection_heatmap_loss"] > 0
    assert ckpt["hparams"]["detection"]["weight"] == 1
    assert not any("detector" in key for key in ckpt["model"])
    samples = sample_batch(loaded, ds.x, ds.mu, ds.sigma, S=2, device="cpu", seed=0)
    assert samples.shape == (2, 3, 16, 16)
    assert run_inference(loaded, ds.x[0], ds.mu, ds.sigma, S=2, device="cpu")[0].shape == (3, 16, 16)
    train(model(), ds, ds, **{**kwargs, "resume": True, "n_epochs": 2})
    with pytest.raises(ValueError, match="Resume detection"):
        train(model(), ds, ds, **{**kwargs, "resume": True, "detection_weight": 0})


def test_joint_cli_and_export_end_to_end(tmp_path, objective):
    import train_joint
    import export_joint
    _, weights = objective
    root, run = tmp_path / "tiles", tmp_path / "run"
    _write_tiles(str(root), _sky(7))
    args = ["--data-root", str(root), "--out-dir", str(run), "--global-batch", "2",
            "--detector-weights", str(weights), "--detection-weight", "1", "--detection-steps", "2",
            "--test-box-px", ",".join(map(str, BOX)), "--patch-size", str(P),
            "--epochs", "1", "--ema-start", "0", "--crops-per-epoch", "2",
            "--base-ch", "8", "--groups", "4", "--time-dim", "32", "--num-workers", "0",
            "--no-early-stop", "--monitor-every", "0", "--skip-eval"]
    train_joint.main(args)
    model_, ckpt = load_checkpoint(run / "best.pt", device="cpu")
    assert model_.model.in_ch == 3 and ckpt["extra"]["joint"]
    assert ckpt["train_detection"] > 0
    out = tmp_path / "export"
    export_joint.main(["--run", str(run), "--data-root", str(root), "--out-dir", str(out),
                       "--S", "2", "--device", "cpu"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["joint"] and manifest["n_windows"] > 0
    for freq in (90, 150, 220):
        pred = next((out / f"f{freq}" / "pred").glob("*.fits"))
        data, header = fits.getdata(pred, header=True)
        assert data.shape == (P, P) and np.isfinite(data).all()
        assert header["BUNIT"] == "uK" and header["FREQ"] == freq
    with pytest.raises(FileExistsError):
        train_joint.main(args)


def _joint_ddp_worker(rank, port, root, weights):
    import torch.distributed as dist
    from bbdm import dist as D
    from bbdm.early_stop import EarlyStopper
    import bbdm.model.detection_loss as module

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    _, _, _, device = D.init_distributed(timeout_minutes=2)
    try:
        module.CrossFrequencyAttentionUNet = TinyCFA
        loss = CFAHeatmapLoss(weights, [10, 20, 30], [50, 60, 70], edge_margin=1)
        ds = SmallDataset()
        bridge, _ = train(model(), ds, ds, n_epochs=2, batch_size=1, ema_start=0,
                          num_workers=0, device=device, checkpoint_dir=str(Path(root) / "ddp"),
                          detection_loss=loss, detection_weight=1, detection_steps=2,
                          monitor_dataset=ds, monitor_n=2, monitor_S=2, monitor_every=1,
                          early_stop=EarlyStopper(tf_max=-1, tf_strikes=1, warmup_checks=0))
        flat = torch.cat([p.detach().flatten() for p in bridge.model.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        assert torch.equal(gathered[0], gathered[1])
    finally:
        D.cleanup()


def test_joint_detection_ddp_and_stop(tmp_path, objective):
    from test_ddp import _free_port
    _, weights = objective
    torch.multiprocessing.spawn(_joint_ddp_worker,
                                 args=(_free_port(), str(tmp_path), str(weights)), nprocs=2, join=True)
    saved = torch.load(tmp_path / "ddp" / "last.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 0 and saved["train_detection"] > 0
    assert saved["monitor"][0]["detection_heatmap_loss"] > 0


def test_full_size_joint_training_refuses_cpu(tmp_path, monkeypatch):
    import train_joint
    monkeypatch.setattr(train_joint.D, "init_distributed", lambda: (0, 1, 0, "cpu"))
    with pytest.raises(ValueError, match="CUDA GPU"):
        train_joint.main(["--data-root", str(tmp_path), "--out-dir", str(tmp_path / "run"),
                          "--global-batch", "1", "--detection-weight", "0"])
