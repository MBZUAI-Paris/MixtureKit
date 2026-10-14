"""Training helpers for models merged with MixtureKit."""

import json
import os

import torch
import transformers

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def load_balancing_weight(labels, num_items_in_batch=None):
    """
    The weight of the load balancing loss of a batch: with gradient accumulation (or
    several devices), transformers divides the language modeling loss of each batch by the
    number of target tokens of all the accumulated batches (`num_items_in_batch`), and the
    load balancing loss gets the share of the target tokens of its batch, so that it is
    averaged over the accumulated batches instead of summed.
    """
    if num_items_in_batch is None:
        return 1.0
    # Counted like transformers counts num_items_in_batch: shifted labels from version 5
    targets = labels[..., 1:] if _TRANSFORMERS_MAJOR >= 5 else labels
    n_tokens = (targets != -100).sum()
    if not isinstance(num_items_in_batch, torch.Tensor):
        num_items_in_batch = torch.tensor(num_items_in_batch)
    # No target token in the accumulated batches: no loss, and no 0 / 0
    return n_tokens / num_items_in_batch.to(n_tokens.device).clamp(min=1)


def moe_loss_function(model):
    """
    A loss function for transformers' Trainer and trl's SFTTrainer (`compute_loss_func`):
    the language modeling loss plus the load balancing loss of the routers, both normalized
    over the accumulated batches. Computing the loss from the model outputs also makes recent
    trl versions call the forward of the merged model, which returns the load balancing loss.

    Parameters
    ----------
    model: PreTrainedModel
        The merged model (or a PEFT model wrapping it).
    """
    vocab_size = model.config.get_text_config().vocab_size

    def loss_function(outputs, labels, num_items_in_batch=None):
        loss = model.loss_function(
            outputs["logits"], labels, vocab_size=vocab_size, num_items_in_batch=num_items_in_batch
        )
        # The unweighted load balancing loss of the batch (the forward has no labels here)
        aux_loss = outputs.get("aux_loss")
        if aux_loss is None:
            return loss
        weight = load_balancing_weight(labels, num_items_in_batch)
        return loss + (aux_loss * weight).to(loss.dtype)

    return loss_function


def mark_trained(output_dir):
    """Records in the saved configuration that the routers or stitch layers were trained."""
    path = os.path.join(output_dir, "config.json")
    if not os.path.exists(path):  # LoRA adapters: the base model's config is unchanged
        return
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config["mixturekit_trained"] = True
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
