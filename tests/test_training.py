"""Training through transformers' Trainer: gradient accumulation, mixed precision, copies."""

import copy

import pytest
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from MixtureKit.training import moe_loss_function
from conftest import PROMPT, load


class RouterGradients(TrainerCallback):
    """Keeps the router gradients of the first optimizer step."""

    grads = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.grads is None:
            self.grads = torch.cat(
                [p.grad.detach().float().flatten().clone() for n, p in model.named_parameters() if n.endswith(".gate.weight")]
            )


def router_gradients(path, tmp_path, alpha, batch_size, accumulation, bf16=False, explicit_loss=False):
    model = load(path)
    model.set_load_balancing_alpha(alpha)
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(".gate.weight"))
    # Two different samples: the accumulated batches see different routing
    samples = [
        {"input_ids": PROMPT[0], "labels": PROMPT[0]},
        {"input_ids": PROMPT[0].flip(0), "labels": PROMPT[0].flip(0)},
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path / f"run_{alpha}_{batch_size}_{bf16}_{explicit_loss}"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_steps=1,
        learning_rate=0.0,
        use_cpu=True,
        bf16=bf16,
        gradient_checkpointing=True,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
    )
    capture = RouterGradients()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=samples,
        callbacks=[capture],
        compute_loss_func=moe_loss_function(model) if explicit_loss else None,
    )
    trainer.train()
    return capture.grads


@pytest.mark.parametrize("explicit_loss", [False, True])
@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_load_balancing_gradient_does_not_depend_on_accumulation(experts, build, tmp_path, method, explicit_loss):
    path = build(method, experts["qwen3"], f"qwen3{method}")

    def aux_part(batch_size, accumulation):
        with_aux = router_gradients(path, tmp_path, 1.0, batch_size, accumulation, explicit_loss=explicit_loss)
        without = router_gradients(path, tmp_path, 0.0, batch_size, accumulation, explicit_loss=explicit_loss)
        return with_aux - without

    one_batch, accumulated = aux_part(2, 1), aux_part(1, 2)
    assert one_batch.abs().sum() > 0
    # Same load balancing gradient, up to the per-batch routing statistics
    ratio = (accumulated.norm() / one_batch.norm()).item()
    assert 0.5 < ratio < 1.5


@pytest.mark.parametrize("explicit_loss", [False, True])
def test_load_balancing_loss_is_kept_in_bf16(experts, build, tmp_path, explicit_loss):
    path = build("btx", experts["qwen3"], "qwen3btx")
    with_aux = router_gradients(path, tmp_path, 1.0, 2, 1, bf16=True, explicit_loss=explicit_loss)
    without = router_gradients(path, tmp_path, 0.0, 2, 1, bf16=True, explicit_loss=explicit_loss)
    assert not torch.allclose(with_aux, without)


@pytest.mark.parametrize("method", ["btx", "traditional", "bts"])
def test_model_can_be_copied_after_a_training_step(experts, build, method):
    model = load(build(method, experts["qwen3"], f"qwen3{method}")).train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model(input_ids=PROMPT, labels=PROMPT).loss.backward()
    twin = copy.deepcopy(model)
    with torch.no_grad():
        assert torch.equal(twin.eval()(input_ids=PROMPT).logits, model.eval()(input_ids=PROMPT).logits)


def test_positional_labels_include_the_load_balancing_loss(experts, build):
    model = load(build("btx", experts["qwen3"], "qwen3btx")).train()
    model.set_load_balancing_alpha(1.0)
    by_name = model(input_ids=PROMPT, labels=PROMPT).loss
    import inspect

    position = list(inspect.signature(type(model).forward).parameters).index("labels") - 1
    args = [None] * (position + 1)
    args[0], args[position] = PROMPT, PROMPT
    assert torch.allclose(model(*args).loss, by_name)


def test_no_target_token_gives_finite_router_gradients(experts, build):
    model = load(build("btx", experts["qwen3"], "qwen3btx")).train()
    model.set_load_balancing_alpha(1.0)
    labels = torch.full_like(PROMPT, -100)
    for loss_function in (None, moe_loss_function(model)):
        model.zero_grad()
        if loss_function is None:
            loss = model(input_ids=PROMPT, labels=labels, num_items_in_batch=torch.tensor(0)).loss
        else:
            loss = loss_function(model(input_ids=PROMPT), labels, num_items_in_batch=torch.tensor(0))
        loss.backward()
        grads = [p.grad for n, p in model.named_parameters() if n.endswith(".gate.weight")]
        assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_aux_loss_output_is_unweighted(experts, build):
    model = load(build("btx", experts["qwen3"], "qwen3btx")).train()
    model.set_load_balancing_alpha(1.0)
    without_labels = model(input_ids=PROMPT).aux_loss
    labels = PROMPT.clone()
    labels[:, :6] = -100
    with_labels = model(input_ids=PROMPT, labels=labels, num_items_in_batch=torch.tensor(100)).aux_loss
    assert torch.allclose(without_labels, with_labels)


@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_load_balancing_loss_leaves_out_padding(experts, build, method):
    """Right padding: the routers see the same real tokens, the loss does not change."""
    model = load(build(method, experts["qwen3"], f"qwen3{method}"))
    model.train()
    padded = torch.cat([PROMPT, torch.zeros(1, 6, dtype=PROMPT.dtype)], dim=1)
    mask = torch.cat([torch.ones_like(PROMPT), torch.zeros(1, 6, dtype=PROMPT.dtype)], dim=1)
    alone = model(input_ids=PROMPT, attention_mask=torch.ones_like(PROMPT)).aux_loss
    with_padding = model(input_ids=padded, attention_mask=mask).aux_loss
    unmasked = model(input_ids=padded).aux_loss
    assert torch.allclose(alone, with_padding, atol=1e-6)
    assert not torch.allclose(alone, unmasked, atol=1e-6)
