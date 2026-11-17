"""Ascend runtime and weight adapters for the official LingBot generator."""
import json
import logging
from pathlib import Path
import torch
import torch_npu
from accelerate import init_empty_weights
from safetensors.torch import load_file


def load_dit(model_cls, checkpoint, kwargs):
    directory = Path(checkpoint) / "transformer"
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    with init_empty_weights():
        model = model_cls(**kwargs)
    expected = set(model.state_dict())
    assert expected == set(index), (sorted(expected - set(index)), sorted(set(index) - expected))
    loaded = set()
    for name in sorted(set(index.values())):
        state = load_file(str(directory / name))
        assert set(state) == {key for key, shard in index.items() if shard == name}
        model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, strict=False, assign=True)
        loaded.update(state)
    assert loaded == expected and all(not p.is_meta for p in model.parameters())
    logging.info("DiT strictly loaded %d tensors from %d shards", len(loaded), len(set(index.values())))
    return model


def load_text_encoder(model, directory):
    """Load Diffusers UMT5 tensors into the original Wan encoder."""
    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    names = {"shared.weight": "token_embedding.weight", "encoder.final_layer_norm.weight": "norm.weight"}
    for i in range(len(model.blocks)):
        source, target = f"encoder.block.{i}", f"blocks.{i}"
        for projection in ("q", "k", "v", "o"):
            names[f"{source}.layer.0.SelfAttention.{projection}.weight"] = f"{target}.attn.{projection}.weight"
        names[f"{source}.layer.0.SelfAttention.relative_attention_bias.weight"] = f"{target}.pos_embedding.embedding.weight"
        names[f"{source}.layer.0.layer_norm.weight"] = f"{target}.norm1.weight"
        names[f"{source}.layer.1.layer_norm.weight"] = f"{target}.norm2.weight"
        for hf, wan in (("wi_0", "gate.0"), ("wi_1", "fc1"), ("wo", "fc2")):
            names[f"{source}.layer.1.DenseReluDense.{hf}.weight"] = f"{target}.ffn.{wan}.weight"
    if set(index) != set(names) or set(model.state_dict()) != set(names.values()):
        raise ValueError("Diffusers UMT5 keys do not match the original Wan encoder.")
    state = {}
    for shard in sorted(set(index.values())):
        tensors = load_file(str(directory / shard))
        if set(tensors) != {key for key, filename in index.items() if filename == shard}:
            raise ValueError(f"Unexpected keys in UMT5 shard {shard}.")
        state.update({names[key]: value for key, value in tensors.items()})
    model.load_state_dict(state, strict=True, assign=True)
    logging.info("Loaded %d Diffusers tensors into the original UMT5 encoder", len(state))


def load_vae(model, path):
    """Invert the Wan 2.1 Diffusers key conversion without modifying tensors."""
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    native = model.state_dict()
    # Use the original key as the value to recover the converter's rename table.
    names = convert_wan_vae_to_diffusers({key: key for key in native})
    tensors = load_file(str(path))
    if set(names) != set(tensors) or set(names.values()) != set(native):
        raise ValueError("Diffusers VAE keys do not match the original Wan 2.1 VAE.")
    model.load_state_dict({names[key]: value for key, value in tensors.items()}, strict=True, assign=True)
    logging.info("Loaded %d Diffusers tensors into the original Wan 2.1 VAE", len(tensors))


def npu_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, dtype):
    if dropout_p or causal or window_size != (-1, -1):
        raise ValueError("The NPU reference supports the dense non-causal Example 03 attention.")
    assert q.shape[0] == 1
    q_len = q.shape[1] if q_lens is None else int(q_lens[0])
    k_len = k.shape[1] if k_lens is None else int(k_lens[0])
    value = v[:, :k_len]
    if value.dtype not in (torch.float16, torch.bfloat16):
        value = value.to(dtype)
    query = q[:, :q_len].to(value.dtype)
    if q_scale is not None:
        query = query * q_scale
    result = torch_npu.npu_fusion_attention(
        query.contiguous(), k[:, :k_len].to(query.dtype).contiguous(),
        value.contiguous(), query.shape[2], "BSND",
        scale=softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5,
    )[0]
    return result.to(q.dtype)
