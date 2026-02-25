import os, json
import streamlit as st
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import pandas as pd

config = {
    "moe_method": "btx",
    "model_type": "nilex",
    "num_experts_per_tok": 2,
    "experts": [
        # {"expert_name": "base_expert", "model_id": "google/gemma-3-4b-it"},
        {
            "expert_name": "expert_1",
            "model_id": "MBZUAI-Paris/Nile-4B-IFT-Arabic-Expert-v2",
        },
        {
            "expert_name": "expert_2",
            "model_id": "MBZUAI-Paris/Nile-4B-IFT-Latin-Expert",
        },
    ],
    "router_layers": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "alpha": 0,
    "router_layers_index": [],
}


def make_unique_names(names):
    seen = {}
    unique = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
    return unique


# local_dir = "./downloaded_checkpoint"

# local_dir = "/home/sagemaker-user/MixtureKit/lm-evaluation-harness-nile-chat-egyptianmmlu/checkpoint-26224"

local_dir = "MBZUAI-Paris/Btx_Model_2Exp"

# local_dir = "models_merge/nilex"


@st.cache_resource
def load_model(model_path: str):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()

    # model     = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()

    model.to("cuda")

    expert_names = [e["model_id"] for e in config["experts"]]
    layer_names = [
        n
        for n, _ in model.named_modules()
        if any(n.endswith(f"{rl}.gate") for rl in config["router_layers"])
    ]
    return tokenizer, model, expert_names, layer_names, config


tokenizer, model, expert_names, layer_names, config = load_model(local_dir)

expert_logits = {}  # {layer: {"indices": [T, K], "weights": [T, K]}}


def make_hook(name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return

        logits = output  # shape: [B, T, E] or [T, E]

        # Use dtype from actual input tensor
        dtype = (
            inputs[0].dtype
            if isinstance(inputs, (tuple, list)) and isinstance(inputs[0], torch.Tensor)
            else torch.float
        )

        weights, selected_experts = torch.topk(
            logits, config["num_experts_per_tok"], dim=-1
        )
        weights = torch.softmax(weights, dim=-1, dtype=torch.float).to(dtype)

        expert_logits[name] = {
            "indices": selected_experts[0],  # [T, K]
            "weights": weights[0],  # [T, K]
        }

    return hook


gate_patterns = [f"{rl}.gate" for rl in config["router_layers"]]
for n, m in model.named_modules():
    if any(p in n for p in gate_patterns):
        m.register_forward_hook(make_hook(n))

st.title("🔀 MoE Token-Routing Visualizer")
user_input = st.text_area("Input text", value="صباح الفل يا صحبي", height=140)

# model.to("cpu")
# inputs.to("cpu")

with st.expander("Layer selection", expanded=False):
    # Initialize session state to track selection
    if "selected_layers" not in st.session_state:
        st.session_state.selected_layers = [layer_names[0]] if layer_names else []

    # Select All button
    if st.button("Select All Layers"):
        st.session_state.selected_layers = layer_names.copy()

    # Clear Selection button (optional, useful UX)
    if st.button("Clear Selection"):
        st.session_state.selected_layers = []

    # Multiselect linked to session state
    chosen_layers = st.multiselect(
        "Choose one *or more* gate layers to visualise",
        options=layer_names,
        default=st.session_state.selected_layers,
        key="selected_layers",
        help="Hold Ctrl / Cmd to pick multiple layers.",
    )

    st.write("*(You can always re-run with a different choice.)*")


if st.button("Run"):
    if not user_input.strip():
        st.warning("Type something first!")
        st.stop()

    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        _ = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    n_experts = len(expert_names)
    n_k = config["num_experts_per_tok"]

    # COARSE-GRAINED
    # st.subheader("📊 Overall routing (all layers)")
    token_weights_sum = defaultdict(lambda: [0.0] * n_experts)
    layer_count = 0

    # DEBUG: Print weights for first token across all layers
    print("🔍 Expert weights for the first token (index 0):\n")

    # Initialize expert-wise weight sums
    # expert_weight_sums = defaultdict(float)
    # layer_count = 0

    # print("🔍 Expert weights for token index 1 (first non-special token):\n")
    token_avg_weights = []

    expert_names = [
        e["model_id"] for e in config["experts"]
    ]  # Use model_id as column names

    # Find non-special token indices
    for i, tok in enumerate(tokens):
        if tok in {tokenizer.bos_token, tokenizer.cls_token}:
            continue

        expert_sums = [0.0] * n_experts
        contributing_layers = 0  # ← FIX

        for layer_name, info in expert_logits.items():
            if "indices" not in info or "weights" not in info:
                continue

            indices = info["indices"].tolist()
            weights = info["weights"].tolist()

            # Skip if token i doesn't exist in this layer
            if i >= len(indices):
                continue

            contributing_layers += 1  # ← FIX

            if config["num_experts_per_tok"] == 1:
                for eid in indices[i]:
                    expert_sums[eid] += 1.0  # ← count, don’t sum probs
            else:
                for eid, ew in zip(indices[i], weights[i]):
                    expert_sums[eid] += ew

        print(contributing_layers)
        # Average only over the contributing layers
        avg_weights = [w / max(1, contributing_layers) for w in expert_sums]
        print(avg_weights)

        token_clean = tok.lstrip("Ġ")
        token_clean = tok.replace("▁", "").replace("_", "").replace("Ġ", "")

        row = [token_clean] + [f"{w*100:.1f}%" for w in avg_weights]
        token_avg_weights.append(row)

    # Convert to DataFrame
    expert_names = make_unique_names(expert_names)

    df = pd.DataFrame(token_avg_weights, columns=["Token"] + expert_names)

    # ADD COLOR-CODED VISUALIZATION FOR OVERALL AVERAGE WHEN n_k == 1
    st.subheader("🎨 Overall Expert Assignment (Color-coded)")

    palette = ["#FFDADA", "#DAFFD8", "#DADAFF", "#FFF8DA", "#FDDADA"]
    legend_items = [
        f"<span style='display:inline-block;background:{palette[i%len(palette)]};"
        f"padding:4px 10px;border-radius:4px;margin-right:4px'>Expert {i}<br/>{name}</span>"
        for i, name in enumerate(expert_names)
    ]
    st.markdown("**Legend:**<br>" + " ".join(legend_items), unsafe_allow_html=True)

    # Extract argmax expert for each token from the average weights
    palette = ["#FFDADA", "#DAFFD8", "#DADAFF", "#FFF8DA", "#FDDADA"]
    colored_tokens = []

    for row in token_avg_weights:
        token_clean = row[0]  # First element is the token
        # Convert percentage strings back to floats to find argmax
        weight_values = [
            float(w.rstrip("%")) for w in row[1:]
        ]  # Remove '%' and convert to float
        argmax_expert = weight_values.index(
            max(weight_values)
        )  # Find index of maximum weight

        # Color the token based on its dominant expert
        color = palette[argmax_expert % len(palette)]
        colored_token = f"<span style='background:{color};padding:2px 4px;border-radius:4px'>{token_clean}</span>"
        colored_tokens.append(colored_token)

    # Display colored tokens
    st.markdown(" ".join(colored_tokens), unsafe_allow_html=True)

    expert_logits_saved = expert_logits.copy()

    ##########################OUTPUT#####################################
    # messages = [{'role': 'user', 'content': user_input}]

    # input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', return_dict=True, add_generation_prompt=True).input_ids.to(model.device)
    # # input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', return_dict=True, add_generation_prompt=True).input_ids.to("cuda")

    # # Generate output
    # # with torch.no_grad():
    # generated = model.language_model.generate(
    #     input_ids,
    #     max_new_tokens=500,  # adjust if you want longer outputs
    #     do_sample=True,
    #     temperature=0.1
    # )

    # # Extract new tokens after the prompt
    # prompt_len = input_ids.shape[1]
    # new_tokens = generated[0][prompt_len:]

    # # Decode only the new tokens
    # output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # # Display
    # st.subheader("📝 Model Output")
    # st.write(output_text)
    ##########################OUTPUT#####################################

    expert_logits = expert_logits_saved

    # Display table in Streamlit
    st.subheader("📊 Average Expert Weights Across All Layers (Per Token)")
    st.table(df)

    # FINE-GRAINED
    st.subheader("🎯 Per-layer routing")
    if not chosen_layers:
        st.info("No layer selected ➜ nothing to display.")
    else:
        for layer in chosen_layers:
            if layer not in expert_logits:
                st.warning(f"Layer “{layer}” is not a router gate or did not fire.")
                continue

            st.markdown(f"**Layer:** `{layer}`")
            info = expert_logits[layer]

            indices = info["indices"].tolist()
            weights = info["weights"].tolist()

            if n_k == 1:
                # Old color-coded visualization
                palette = ["#FFDADA", "#DAFFD8", "#DADAFF", "#FFF8DA", "#FDDADA"]
                cleaned_tokens, cleaned_assigns = [], []
                for tok, ex in zip(tokens, [row[0] for row in indices]):
                    if tok in {tokenizer.bos_token, tokenizer.cls_token}:
                        continue
                    cleaned_tokens.append(tok.lstrip("Ġ"))
                    cleaned_assigns.append(ex)

                def colored(tok, ex):
                    c = palette[ex % len(palette)]
                    return f"<span style='background:{c};padding:2px 4px;border-radius:4px'>{tok}</span>"

                st.markdown(
                    " ".join(
                        colored(t, e) for t, e in zip(cleaned_tokens, cleaned_assigns)
                    ),
                    unsafe_allow_html=True,
                )
            else:
                # Show table of top-k expert weights per token
                rows = []
                # expert_cols = [f"Expert {i}" for i in range(n_experts)]

                for tok, top_ids, top_ws in zip(tokens, indices, weights):
                    if tok in {tokenizer.bos_token, tokenizer.cls_token}:
                        continue

                    tok_clean = tok.lstrip("Ġ")
                    tok_clean = tok.replace("▁", "").replace("_", "").replace("Ġ", "")

                    # Build mapping: expert ID → weight
                    expert_weight_map = {eid: ew for eid, ew in zip(top_ids, top_ws)}

                    # Create row in fixed expert column order
                    row = [tok_clean] + [
                        f"{expert_weight_map.get(eid, 0.0)*100:5.1f}%"
                        for eid in range(n_experts)
                    ]
                    rows.append(row)

                # Build table
                df = pd.DataFrame(rows, columns=["Token"] + expert_names)
                st.dataframe(df)
