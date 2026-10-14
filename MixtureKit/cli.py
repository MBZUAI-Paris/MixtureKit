"""
The `mixturekit` command.

    mixturekit build config.yaml [--output DIR] [--push-to-hub ORG/NAME]
    mixturekit train config_training.yaml
    mixturekit visualize MODEL [--text "..." --html routing.html]
    mixturekit card MODEL [--repo-id ORG/NAME]
    mixturekit push MODEL ORG/NAME [--private]
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys


def _visualize(args):
    if args.text is None and args.html is None:
        if importlib.util.find_spec("streamlit") is None:
            sys.exit("The visualizer app needs streamlit: pip install 'mixturekit[vis]'")
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer_app.py")
        command = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(args.port), "--"]
        command += [args.model] + (["--device", args.device] if args.device else [])
        return subprocess.call(command)

    from .routing import load_model, trace_routing

    model, tokenizer = load_model(args.model, device=args.device)
    text = args.text if args.text is not None else DEFAULT_TEXT
    trace = trace_routing(model, tokenizer, text)
    share = trace.expert_share()
    print(f"{trace.method.upper()} · {len(trace.tokens)} tokens · {len(trace.layers)} layers")
    for name, fraction in zip(trace.experts, share):
        print(f"  {fraction:6.1%}  {name}")
    path = args.html or "routing.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Token routing</title>")
        f.write(trace.to_html(title=f"Token routing · {args.model}"))
    print(f"written to {path}")
    return 0


DEFAULT_TEXT = "Bonjour ! The integral of x^2 is x^3 / 3. صباح الخير يا صاحبي"

DTYPES = ("bfloat16", "float16", "float32")


def _build(args):
    import torch
    import yaml

    from . import build_moe

    if not os.path.isfile(args.config):
        sys.exit(f"{args.config}: no such file")
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    missing = [key for key in ("moe_method", "model_type", "experts") if not isinstance(config, dict) or key not in config]
    if missing:
        sys.exit(f"{args.config}: missing {', '.join(missing)} (see examples/configs)")
    config.setdefault("router_layers", [])
    config.setdefault("router_layers_index", [])
    config.setdefault("num_experts_per_tok", 2)
    # Options of the build itself may sit in the file; the command line wins
    file_dtype, file_output = config.pop("dtype", None), config.pop("output_dir", None)
    file_streaming = config.pop("streaming", True)
    if not isinstance(file_streaming, bool):
        sys.exit(f"{args.config}: streaming must be true or false")
    dtype = args.dtype or file_dtype or "bfloat16"
    output = args.output or file_output
    streaming = file_streaming and not args.no_streaming
    if dtype not in DTYPES:
        sys.exit(f"dtype must be one of {', '.join(DTYPES)}")
    path = build_moe(config, torch_dtype=getattr(torch, dtype), streaming=streaming, output_dir=output)
    # The configuration it was built from, with the datatype actually used
    saved = os.path.join(path, "mixturekit_config.json")
    with open(saved, encoding="utf-8") as f:
        build_config = json.load(f)
    build_config["dtype"] = dtype
    with open(saved, "w", encoding="utf-8") as f:
        json.dump(build_config, f, indent=2)
    print(f"merged model: {path}")
    if args.push_to_hub:
        from .hub import push_to_hub

        push_to_hub(path, args.push_to_hub, private=args.private)
        print(f"uploaded to https://huggingface.co/{args.push_to_hub}")
    return 0


def _train(args):
    if not os.path.isfile(args.config):
        sys.exit(f"{args.config}: no such file")
    if importlib.util.find_spec("trl") is None:
        sys.exit("Training needs the training dependencies: pip install 'mixturekit[train]'")
    from .train import main as train

    train(args.config)
    return 0


def _card(args):
    from .hub import write_model_card

    try:
        print(write_model_card(args.model, repo_id=args.repo_id, overwrite=True))
    except ValueError as error:
        sys.exit(str(error))
    return 0


def _push(args):
    from .hub import push_to_hub

    try:
        push_to_hub(args.model, args.repo_id, private=args.private)
    except ValueError as error:
        sys.exit(str(error))
    print(f"uploaded to https://huggingface.co/{args.repo_id}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mixturekit", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="merge experts into a Mixture-of-Experts")
    build.add_argument("config", help="a YAML file with the build_moe configuration")
    build.add_argument("--output", help="the output directory (default: models_merge/<model_type>)")
    build.add_argument("--dtype", choices=DTYPES, help="weights datatype (default: bfloat16)")
    build.add_argument("--no-streaming", action="store_true", help="load the full experts")
    build.add_argument("--push-to-hub", metavar="ORG/NAME", help="upload the merged model with its card")
    build.add_argument("--private", action="store_true", help="create the Hub repository as private")
    build.set_defaults(handler=_build)

    train = commands.add_parser("train", help="train the routers (or stitch layers) of a merged model")
    train.add_argument("config", help="a YAML training configuration (see config_training.yaml)")
    train.set_defaults(handler=_train)

    visualize = commands.add_parser(
        "visualize", help="see which experts every token is routed to"
    )
    visualize.add_argument("model", help="a local directory or a Hub id of a merged model")
    visualize.add_argument("--text", help="route this text and write an HTML view instead of opening the app")
    visualize.add_argument("--html", help="where to write the HTML view (default: routing.html)")
    visualize.add_argument("--device", help="cuda, mps or cpu (default: the first available)")
    visualize.add_argument("--port", type=int, default=8501, help="port of the app (default: 8501)")
    visualize.set_defaults(handler=_visualize)

    card = commands.add_parser("card", help="write the model card (README.md) of a merged model")
    card.add_argument("model", help="the directory of the merged model")
    card.add_argument("--repo-id", help="its Hub id, used in the card's snippets")
    card.set_defaults(handler=_card)

    push = commands.add_parser("push", help="upload a merged model and its card to the Hub")
    push.add_argument("model", help="the directory of the merged model")
    push.add_argument("repo_id", metavar="ORG/NAME", help="the Hub repository")
    push.add_argument("--private", action="store_true", help="create the repository as private")
    push.set_defaults(handler=_push)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
