# API

## Building

```{eval-rst}
.. autofunction:: MixtureKit.build_moe

.. autofunction:: MixtureKit.upgrade_checkpoint
```

## Routing

```{eval-rst}
.. automodule:: MixtureKit.routing
   :members: load_model, trace_routing, RoutingTrace, expert_names, available_devices, default_device
```

## Model cards and the Hub

```{eval-rst}
.. automodule:: MixtureKit.hub
   :members: model_card, write_model_card, push_to_hub
```

## Export

```{eval-rst}
.. automodule:: MixtureKit.export
   :members: export_native, native_config
```

## Training

```{eval-rst}
.. automodule:: MixtureKit.training
   :members: moe_loss_function, load_balancing_weight, mark_trained
```
