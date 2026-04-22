"""Interactive YAML config selection (InquirerPy fuzzy UI: filterable list, bordered, fzf-style)."""

import os
import sys
from pathlib import Path


def _list_yaml_configs(config_dir):
    root = Path(config_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Config directory not found: {root}")
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    )
    return files


def _gatekeeper_style():
    """Dark palette similar to modern agent CLIs (subtle frame, soft violet accents)."""
    from InquirerPy.utils import get_style

    return get_style(
        {
            "questionmark": "#52525b",
            "question": "#fafafa bold",
            "answer": "#c4b5fd bold",
            "pointer": "#818cf8 bold",
            "highlighted": "bg:#27272a #f4f4f5",
            "selected": "#a1a1aa",
            "instruction": "#71717a",
            "long_instruction": "#52525b",
            "fuzzy_prompt": "#e4e4e7 bold",
            "fuzzy_info": "#71717a",
            "fuzzy_border": "#3f3f46",
            "fuzzy_match": "#a78bfa bold",
            "input": "#fafafa",
        },
        style_override=False,
    )


def _pick_config_fuzzy(files, purpose, config_dir):
    """InquirerPy fuzzy prompt: type-to-filter, bordered list, fzf-like."""
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError as e:
        raise ImportError(
            "Interactive config selection requires InquirerPy. "
            "Install with: pip install InquirerPy\n"
            "Or pass -c / --config with a path to your YAML (e.g. -c tracking.yaml)."
        ) from e

    folder_hint = Path(config_dir).name
    choices = [
        Choice(
            value=str(p.resolve()),
            name=f"{p.name}",
        )
        for p in files
    ]
    header = f"gatetracker  ·  {folder_hint}/  ·  {purpose}"
    try:
        selected = inquirer.fuzzy(
            message=header,
            choices=choices,
            default="",
            pointer=" ❯ ",
            qmark="",
            amark="",
            prompt=" ",
            style=_gatekeeper_style(),
            border=True,
            info=True,
            match_exact=False,
            max_height="55%",
            instruction="filter",
            long_instruction="↑↓ navigate  ·  type to narrow  ·  enter confirm  ·  ctrl+c cancel",
            mandatory=True,
            cycle=True,
            wrap_lines=True,
        ).execute()
    except KeyboardInterrupt:
        sys.stderr.write("\nCancelled.\n")
        raise SystemExit(130) from None

    if not selected:
        sys.stderr.write("No configuration selected.\n")
        raise SystemExit(1)
    return selected


def normalize_user_config_path(repo_root, user_path):
    """Resolve ``user_path`` to an existing YAML file (``config/`` prefix, repo-relative, or absolute)."""
    if not user_path:
        return None
    user_path = os.path.expanduser(user_path.strip())
    if os.path.isfile(user_path):
        return os.path.normpath(os.path.abspath(user_path))
    candidate = os.path.join(repo_root, "config", user_path)
    if os.path.isfile(candidate):
        return os.path.normpath(os.path.abspath(candidate))
    candidate2 = os.path.join(repo_root, user_path)
    if os.path.isfile(candidate2):
        return os.path.normpath(os.path.abspath(candidate2))
    raise FileNotFoundError(
        f"Config file not found: {user_path!r} (tried config/ and repo root)."
    )


def resolve_config_yaml_path_interactive(config_dir, purpose):
    """
    If stdin is a TTY, show an InquirerPy fuzzy selector; otherwise print paths and exit.

    Returns:
        Absolute path to the chosen YAML file.
    """
    files = _list_yaml_configs(config_dir)
    if not files:
        print(
            f"\n[GateTracker] No .yaml/.yml files found in `{config_dir}`.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not sys.stdin.isatty():
        print(
            "\nNot a TTY — cannot open the interactive config picker. "
            "Pass -c / --config with a path under config/.\n\n"
            f"Available in {config_dir}:",
            file=sys.stderr,
        )
        for p in files:
            print(f"  - {p.name}", file=sys.stderr)
        sys.exit(1)

    resolved = _pick_config_fuzzy(files, purpose, config_dir)
    return resolved
