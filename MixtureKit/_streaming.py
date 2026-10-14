"""
Memory-light composition. The experts are read tensor by tensor from their
safetensors files and the merged checkpoint is written tensor by tensor, so the
peak memory stays around the size of the largest tensor instead of whole models.
"""

import json
import os
import re
import shutil
import struct

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig

_TO_SAFETENSORS = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in [
    ("uint16", "U16"),
    ("uint32", "U32"),
    ("uint64", "U64"),
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
]:
    if hasattr(torch, _name):
        _TO_SAFETENSORS[getattr(torch, _name)] = _code
_FROM_SAFETENSORS = {name: dtype for dtype, name in _TO_SAFETENSORS.items()}


class NotStreamable(Exception):
    """The expert checkpoint cannot be mapped tensor by tensor onto the model parameters."""


def _instantiate_on_meta(model_cls, config, torch_dtype):
    """
    This function builds the model skeleton without allocating any weight.
    """
    with torch.device("meta"):
        if hasattr(model_cls, "from_config"):
            return model_cls.from_config(
                config, trust_remote_code=True, torch_dtype=torch_dtype
            )
        return model_cls._from_config(config, torch_dtype=torch_dtype)


class ExpertCheckpoint:
    def __init__(self, model_id, model_cls, torch_dtype):
        """
        A lazy, read-only view of the weights of an expert, keyed by the
        parameter names of the model (as in `model.state_dict()`).

        Parameters
        ----------
        model_id: str
            The id of the expert on the HF Hub or a local directory.
        model_cls: type
            The class used to load the expert.
        torch_dtype: torch.dtype
            The datatype of the floating point weights.

        Raises
        ------
        NotStreamable
            When the safetensors files do not match the model parameters.
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        try:
            # Same configuration class as from_pretrained, e.g. the text config of a
            # multimodal checkpoint when the model class is a text-only causal LM
            config_cls = getattr(model_cls, "config_class", None)
            if config_cls is None:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            else:
                config = config_cls.from_pretrained(model_id)
            model = _instantiate_on_meta(model_cls, config, torch_dtype)
        except Exception as error:
            raise NotStreamable(f"cannot build the model skeleton: {error}") from error

        if getattr(model, "_keep_in_fp32_modules", None):
            raise NotStreamable(f"{type(model).__name__} keeps modules in float32")

        self.tied_keys = list(getattr(model, "_tied_weights_keys", None) or [])
        self.config = model.config
        self.config.torch_dtype = torch_dtype
        self.generation_config = self._generation_config(model_id, model)

        expected = model.state_dict(keep_vars=True)
        self._directory = self._local_directory(model_id)
        self._handles = {}
        stored = self._stored_tensors(model)
        del model

        # Tied parameters share the same object, e.g. lm_head and embed_tokens
        owners = {}
        for name, tensor in expected.items():
            owners.setdefault(id(tensor), []).append(name)

        # A tied copy always takes the values of the owner of the shared parameter,
        # as the loaded model does, even when the checkpoint also stores the copy
        self._sources = {}
        self.aliases = set()
        for name, tensor in expected.items():
            group = owners[id(tensor)]
            owner = next((n for n in group if n in stored), None)
            if owner is None:
                raise NotStreamable(f"'{name}' is not stored in {model_id}")
            self._sources[name] = stored[owner]
            if name != owner:
                self.aliases.add(name)

            file, key = self._sources[name]
            stored_dtype = self._handle(file).get_slice(key).get_dtype()
            if stored_dtype not in _FROM_SAFETENSORS:
                raise NotStreamable(f"unsupported dtype {stored_dtype} for '{name}'")
            shape = tuple(self._handle(file).get_slice(key).get_shape())
            if shape != tuple(tensor.shape):
                raise NotStreamable(
                    f"'{name}' has shape {shape} in {model_id}, expected {tuple(tensor.shape)}"
                )

    @staticmethod
    def _generation_config(model_id, model):
        try:
            return GenerationConfig.from_pretrained(model_id).to_dict()
        except OSError:
            return model.generation_config.to_dict()

    @staticmethod
    def _local_directory(model_id):
        if os.path.isdir(model_id):
            return model_id
        # Download only the shards used by the model, not other weight formats of the repo
        try:
            index = hf_hub_download(model_id, "model.safetensors.index.json")
            with open(index, "r", encoding="utf-8") as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        except EntryNotFoundError:
            files = ["model.safetensors"]
        return snapshot_download(model_id, allow_patterns=["*.json", *files])

    def _handle(self, file):
        if file not in self._handles:
            path = os.path.join(self._directory, file)
            self._handles[file] = safe_open(path, framework="pt", device="cpu")
        return self._handles[file]

    def _stored_tensors(self, model):
        """
        This function maps the parameter names to their (file, key) in the checkpoint,
        renaming the keys as `from_pretrained` does.
        """
        index = os.path.join(self._directory, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        elif os.path.isfile(os.path.join(self._directory, "model.safetensors")):
            keys = self._handle("model.safetensors").keys()
            weight_map = {key: "model.safetensors" for key in keys}
        else:
            raise NotStreamable(f"no safetensors weights found for {self.model_id}")

        key_mapping = getattr(model, "_checkpoint_conversion_mapping", None) or {}
        # A task model (e.g. a causal LM) loaded from a checkpoint that nests it under
        # the base model prefix, e.g. "language_model." in multimodal Gemma3 checkpoints
        prefix = getattr(model, "base_model_prefix", "")
        has_prefix = bool(prefix) and any(key.startswith(prefix) for key in weight_map)
        expects_prefix = bool(prefix) and hasattr(model, prefix)

        stored = {}
        for key, file in weight_map.items():
            name, _ = model._fix_state_dict_key_on_load(key)
            for pattern, replacement in key_mapping.items():
                name, n_replaced = re.subn(pattern, replacement, name)
                if n_replaced:
                    break
            if expects_prefix and not has_prefix:
                name = f"{prefix}.{name}"
            elif has_prefix and not expects_prefix:
                if not name.startswith(f"{prefix}."):
                    continue
                name = name[len(prefix) + 1 :]
            stored[name] = (file, key)
        return stored

    def __contains__(self, name):
        return name in self._sources

    def names(self):
        return list(self._sources)

    def shape(self, name):
        file, key = self._sources[name]
        return tuple(self._handle(file).get_slice(key).get_shape())

    def dtype(self, name):
        """The datatype of the parameter once loaded with `torch_dtype`."""
        file, key = self._sources[name]
        stored = _FROM_SAFETENSORS[self._handle(file).get_slice(key).get_dtype()]
        return self.torch_dtype if stored.is_floating_point else stored

    def tensor(self, name):
        file, key = self._sources[name]
        tensor = self._handle(file).get_tensor(key)
        if tensor.is_floating_point():
            tensor = tensor.to(self.torch_dtype)
        return tensor


def _parse_size(size):
    """Converts "9GB" / "500MiB" / 1000 to a number of bytes."""
    if isinstance(size, int):
        return size
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(I?)B\s*", size.upper())
    if match is None:
        raise ValueError(f"Invalid shard size: {size}")
    base = 1024 if match.group(3) else 1000
    return int(float(match.group(1)) * base ** " KMGT".index(match.group(2) or " "))


def write_safetensors(directory, entries, max_shard_size="9GB"):
    """
    This function writes the merged checkpoint without holding it in memory.
    The headers are built from the planned shapes, then each tensor is computed
    and written right away.

    Parameters
    ----------
    directory: str
        The output directory.
    entries: list
        Tuples (name, shape, dtype, compute) where compute() returns the tensor.
    max_shard_size: int or str, default="9GB"
        The largest size of a safetensors shard.
    """
    limit = _parse_size(max_shard_size)
    for _, _, dtype, _ in entries:
        if dtype not in _TO_SAFETENSORS:
            raise ValueError(f"Cannot write {dtype} tensors to safetensors")

    # The new shards are written apart and replace the previous ones once complete
    staging = os.path.join(directory, ".mixturekit-staging")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    nbytes = lambda shape, dtype: torch.Size(shape).numel() * dtype.itemsize

    shards, current, current_size = [], [], 0
    for entry in entries:
        size = nbytes(entry[1], entry[2])
        if current and current_size + size > limit:
            shards.append(current)
            current, current_size = [], 0
        current.append(entry)
        current_size += size
    shards.append(current)

    n_shards = len(shards)
    weight_map, total_size = {}, 0
    for shard_idx, shard in enumerate(shards, start=1):
        file = (
            "model.safetensors"
            if n_shards == 1
            else f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
        )
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        for name, shape, dtype, _ in shard:
            size = nbytes(shape, dtype)
            header[name] = {
                "dtype": _TO_SAFETENSORS[dtype],
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size
            weight_map[name] = file
        total_size += offset

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_bytes += b" " * (-len(header_bytes) % 8)

        with open(os.path.join(staging, file), "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for name, shape, dtype, compute in shard:
                tensor = compute()
                if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                    raise RuntimeError(
                        f"'{name}' was planned as {tuple(shape)} {dtype}, "
                        f"got {tuple(tensor.shape)} {tensor.dtype}"
                    )
                data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                f.write(data.numpy().data)
                del tensor, data

    if n_shards > 1:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(staging, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    # Shards of a previous build would be loaded instead of (or mixed with) the new ones
    for file in os.listdir(directory):
        if file == "model.safetensors.index.json" or re.fullmatch(
            r"model(-\d{5}-of-\d{5})?\.safetensors", file
        ):
            os.remove(os.path.join(directory, file))
    for file in os.listdir(staging):
        os.replace(os.path.join(staging, file), os.path.join(directory, file))
    os.rmdir(staging)
