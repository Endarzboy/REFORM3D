"""CPU inference must match float32 encoder outputs without CUDA autocast."""

from dataclasses import replace

import pytest
import torch

from reform3d.encoders.med3dino.vision_transformer import DinoVisionTransformer3d
from reform3d.modules.acquisition_robustness import feature_extractor as runtime


@pytest.mark.parametrize("multilayer", [False, True])
@pytest.mark.parametrize("device_override", [False, True])
def test_cpu_extractor_matches_float32_encoder(tmp_path, monkeypatch, multilayer, device_override):
    def tiny_model():
        return DinoVisionTransformer3d(
            img_size=8, patch_size=4, in_chans=1, embed_dim=16,
            depth=2, num_heads=2, block_chunks=1, init_values=1e-5,
        ).eval()

    torch.manual_seed(21)
    reference = tiny_model()
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"teacher": reference.state_dict()}, checkpoint)
    registry_config = replace(runtime.resolve_checkpoint_config("med3dino_rel_96"), crop_size=8)
    monkeypatch.setattr(runtime, "resolve_checkpoint_config", lambda name: registry_config)

    def build_model(self):
        model = tiny_model()
        self.embed_dim = model.embed_dim
        self.n_blocks = model.n_blocks
        return model

    def reject_cuda_autocast(*args, **kwargs):
        raise AssertionError("CPU inference must not enter CUDA autocast")

    monkeypatch.setattr(runtime.FeatureExtractor, "_build_model", build_model)
    monkeypatch.setattr(torch.cuda.amp, "autocast", reject_cuda_autocast)
    config = runtime.ExtractionConfig(
        device="cuda" if device_override else "cpu",
        extract_multilayer=multilayer, n_last_layers=2,
    )
    if not device_override:
        assert config.dtype == torch.float32
    extractor = runtime.FeatureExtractor(
        "med3dino_rel_96", config=config,
        device="cpu" if device_override else None, checkpoint_path=checkpoint,
    )
    assert extractor.config.dtype == torch.float32
    assert next(extractor.model.parameters()).dtype == torch.float32
    volume = torch.randn(1, 1, 8, 8, 8).to(torch.float16)
    with torch.inference_mode():
        expected = reference.forward_features(volume.float())
        actual = extractor.extract(volume, normalize=False)
    torch.testing.assert_close(actual["cls"], expected["x_norm_clstoken"], atol=0, rtol=0)
    torch.testing.assert_close(
        actual["avg_pool"], expected["x_norm_patchtokens"].mean(dim=1), atol=0, rtol=0,
    )
    if multilayer:
        with torch.inference_mode():
            layers = reference.get_intermediate_layers(volume.float(), n=2, return_class_token=True)
        expected_multilayer = torch.cat([cls for _, cls in layers], dim=-1)
        torch.testing.assert_close(actual["multilayer"], expected_multilayer, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Expected 5D input"):
        extractor.extract(volume.squeeze(0))


def test_cuda_configuration_retains_mixed_precision():
    config = runtime.ExtractionConfig(device="cuda:0")
    assert config.dtype == torch.float16
