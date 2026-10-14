# Which method?

MixtureKit composes experts in three ways. All three keep the experts' weights and add
small trainable parts (routers or stitch layers).

| | BTX | Traditional | BTS |
| --- | --- | --- | --- |
| What is kept per expert | the projections you choose (e.g. the MLP) | the feed-forward block | the whole model |
| What is shared | everything else, averaged | everything else, averaged | nothing: the experts run next to a hub |
| Trained | one router per routed projection | one router per layer | stitch layers |
| Compute per token | top-k of the routed projections | top-k feed-forward blocks | all the experts |
| Size | grows with the routed projections | close to a Mixtral of the experts | sum of the experts |
| Good for | fine-tunes of one base, with a choice of what to route | a classic sparse MoE | experts that diverged a lot: no weight is averaged |

```{tip}
Start with **traditional** for fine-tunes of the same base model. Use **BTX** to route
other projections too (e.g. attention), and **BTS** when averaging the shared weights hurts:
experts trained far apart from each other.
```

## BTX: Branch-Train-MiX

[Branch-Train-MiX](https://arxiv.org/abs/2403.07816) keeps, for every projection listed in
`router_layers` (e.g. `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `self_attn.q_proj`), one
copy per expert behind its own top-k router (`linear_attn` projections of Qwen3.5 can be routed too). The other weights (embeddings, norms, the other
projections) are averaged over the experts.

## Traditional: Mixtral-style MoE

Each decoder block gets one router over the experts' feed-forward blocks, the layout of
Mixtral (`mlp.gate` and `mlp.experts.N`). The other weights are averaged.

## BTS: Branch-Train-Stitch

[Branch-Train-Stitch](https://arxiv.org/abs/2502.00075) keeps every expert intact. The first
expert is the **hub**; the other experts run in parallel streams. Every `stitch_freq` blocks
a stitch layer exchanges information between the streams, alternately from the hub into the
experts and from the experts into the hub; the last block always merges the experts into the
hub. The projections start as identities and the gates at zero: a merging stitch starts as
the plain average of the streams, a hub-to-expert stitch as an even mix of the expert and the
hub. Training the stitches learns how much of each stream to take.

## The load balancing loss

BTX and traditional routers are trained with the Switch Transformer load balancing loss,
averaged over the routers and normalized to equal 1 for a perfectly balanced load, so
`alpha` has the same scale whatever the depth (`0.01` is a common choice). It is added to the
language modeling loss during training; `outputs.aux_loss` holds it, already multiplied by
`alpha` (in training mode, when `alpha > 0`).
