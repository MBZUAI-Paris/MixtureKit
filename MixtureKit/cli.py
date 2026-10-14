"""
The `mixturekit` command.

    mixturekit visualize MODEL                      # opens the routing visualizer
    mixturekit visualize MODEL --text "..." --html routing.html
"""

import argparse
import importlib.util
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mixturekit", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    visualize = commands.add_parser(
        "visualize", help="see which experts every token is routed to"
    )
    visualize.add_argument("model", help="a local directory or a Hub id of a merged model")
    visualize.add_argument("--text", help="route this text and write an HTML view instead of opening the app")
    visualize.add_argument("--html", help="where to write the HTML view (default: routing.html)")
    visualize.add_argument("--device", help="cuda, mps or cpu (default: the first available)")
    visualize.add_argument("--port", type=int, default=8501, help="port of the app (default: 8501)")
    visualize.set_defaults(handler=_visualize)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
