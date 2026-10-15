"""
The routing visualizer, a Streamlit app.

    mixturekit visualize MODEL
    streamlit run MixtureKit/visualizer_app.py -- MODEL [--device cpu]
"""

import argparse
import os
import sys

import streamlit as st

from MixtureKit.cli import DEFAULT_TEXT
from MixtureKit.routing import PALETTE, available_devices, default_device, load_model, trace_routing


def _arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--device", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    # The environment wins and fixes the model, e.g. in a Hugging Face Space:
    # MIXTUREKIT_MODEL=org/model (a merged model runs its own modeling code, so visitors of a
    # public app must not choose it)
    args.fixed = bool(os.environ.get("MIXTUREKIT_MODEL"))
    args.model = os.environ.get("MIXTUREKIT_MODEL") or args.model
    return args


@st.cache_resource(show_spinner="Loading the model...")
def _load(model_id, device):
    return load_model(model_id, device=device)


def main():
    st.set_page_config(page_title="MixtureKit routing", layout="wide")
    args = _arguments()

    with st.sidebar:
        st.header("Model")
        model_id = st.text_input("Merged model (directory or Hub id)", value=args.model, disabled=args.fixed)
        devices = available_devices()
        if args.device and args.device not in devices:
            devices.insert(0, args.device)
        device = st.selectbox("Device", devices, index=devices.index(args.device or default_device()))

    st.title("Which expert does every token use?")
    if not model_id:
        st.info("Enter a model merged with MixtureKit in the sidebar.")
        return
    try:
        model, tokenizer = _load(model_id, device)
    except Exception as error:  # shown in the app instead of a stack trace
        st.error(f"Cannot load {model_id} on {device}: {error}")
        return

    # A link can carry the text: ?text=...
    initial = st.query_params.get("text") or DEFAULT_TEXT
    limit = 2000 if args.fixed else None
    text = st.text_area("Text", value=initial[:limit] if limit else initial, height=120, max_chars=limit)
    if not text.strip():
        st.warning("Type some text to route.")
        return
    trace = trace_routing(model, tokenizer, text)
    layer_names = list(trace.layers)

    with st.sidebar:
        st.header("Layers")
        chosen = st.multiselect(
            "Average over (all by default)", layer_names, default=[],
            help="Pick layers to see their routing alone.",
        )
    layers = chosen or None

    st.html(trace.to_html(layers=layers, title="Main expert of every token"))

    import pandas as pd

    weights = trace.mean_weights(layers)
    table = pd.DataFrame(
        (weights * 100).round(decimals=1).tolist(), columns=trace.experts,
        index=[f"{i}: {t!r}" for i, t in enumerate(trace.tokens)],
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Expert weights per token (%)")
        st.dataframe(table, width="stretch")
    with right:
        st.subheader("Load of every expert per layer")
        # Numbered labels, padded: the chart sorts its categories as text
        width = len(str(len(layer_names) - 1))
        order = [f"{i:0{width}d} · {name}" for i, name in enumerate(layer_names)]
        load = pd.DataFrame(trace.layer_load().tolist(), columns=trace.experts, index=order)
        # The colours of the experts in the token view
        colours = [PALETTE[i % len(PALETTE)] for i in range(len(trace.experts))]
        st.bar_chart(load, horizontal=True, stack=True, color=colours)
        st.caption(
            "Mean weight of each expert over the tokens. A layer where one expert takes "
            "everything has collapsed routing."
        )


if __name__ == "__main__":
    main()
