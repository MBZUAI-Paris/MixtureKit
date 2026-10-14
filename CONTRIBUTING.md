# Contributing to MixtureKit

Thanks for helping! Bug reports, new model families, docs fixes and ideas are all welcome.

## Report a bug or ask for a feature

Open an [issue](https://github.com/MBZUAI-Paris/MixtureKit/issues/new/choose). For a bug,
include the versions (`pip show mixturekit transformers torch`), the build configuration
(YAML) and the full error. A failing case with tiny models (see `tests/conftest.py`) is the
fastest way to a fix.

For larger changes (a new composition method, a new command), open an issue first so we can
agree on the design.

## Set up

```bash
git clone https://github.com/MBZUAI-Paris/MixtureKit && cd MixtureKit
pip install -e ".[test]"
pytest                      # CPU, tiny models built on the fly, no download, a few minutes
```

The docs:

```bash
pip install -e ".[doc]"
sphinx-build -W docs docs/_build/html
```

## Pull requests

- One change per pull request, with a test that fails without it.
- The tests run on CPU with tiny models from `tests/conftest.py`: no download, no GPU. A new
  model family gets a tiny config in `family_config` and runs every method.
- CI tests transformers 4.52 to 5.19 (a few versions in between): code that depends on the
  transformers version must work on all of them.
- Merged models must keep loading without MixtureKit installed:
  `MixtureKit/modeling_template.py` only uses torch and transformers.
- Update the docs page (`docs/`) of what you change.

## Good to know

- `MixtureKit/MoeExperts.py` composes the experts (`build_moe`); `_streaming.py` reads and
  writes them tensor by tensor.
- `MixtureKit/modeling_template.py` is written next to the merged weights: the model
  subclasses the transformers class of the experts and converts its modules after creation.
- `MixtureKit/train.py` is the training script behind `mixturekit train`.

By contributing, you agree that your contributions are released under the
[BSD 3-Clause License](LICENSE).
