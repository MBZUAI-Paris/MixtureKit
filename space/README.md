---
title: MixtureKit Routing Visualizer
emoji: 🔀
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8501
license: bsd-3-clause
short_description: See which expert every token uses in a MixtureKit MoE
tags:
  - mixturekit
  - mixture-of-experts
  - moe
  - visualization
---

# MixtureKit routing visualizer

Type a text and see which expert of a Mixture-of-Experts model built with
[MixtureKit](https://github.com/MBZUAI-Paris/MixtureKit) every token is sent to, overall and
layer by layer.

The model shown is set by the `MIXTUREKIT_MODEL` variable of this Space.
