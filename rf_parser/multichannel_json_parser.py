#!/usr/bin/env python3
"""
multichannel_json_parser.py  –  Single-channel JSON → multi-channel JSON converter

Takes a single-channel pulse sequence JSON (with a flat "timeline" list) and
produces a multi-channel JSON where "timeline" is a dict of N independent
channel timelines.  All other top-level keys (e.g. "metadata", "waveforms")
are preserved verbatim.

Each timeline entry in the output gains two fields:
    gain   (float, default 1)  – magnitude scale relative to the waveform entry
    phase  (float, default 0)  – phase offset in radians relative to the waveform entry

Usage
-----
    python multichannel_json_parser.py <input.json> [options]

Options
-------
    -o / --output FILE          Write JSON to FILE  [default: stdout]
    --n-channels N              Number of output channels  [default: 8]
    --gains  0:1.0,1:0.5   Uniform gain per channel (comma-separated key:value pairs)
    --phases 1:1.5708         Uniform phase (rad) per channel

Python API
----------
    from multichannel_json_parser import parse_multichannel

    result = parse_multichannel(
        input_json,
        n_channels=8,
        channel_gains={"ch1": 1.0, "ch2": 0.5},
        channel_phases={"ch2": 1.5708},
    )
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


def parse_multichannel(
    input_json: dict,
    *,
    n_channels: int = 8,
    channel_gains: Optional[Dict[str, Union[float, List[float]]]] = None,
    channel_phases: Optional[Dict[str, Union[float, List[float]]]] = None,
) -> dict:
    """
    Convert a single-channel sequence JSON to a multi-channel JSON.

    Parameters
    ----------
    input_json : dict
        Parsed JSON with at least a "timeline" key whose value is a list of
        event dicts.  All other keys are copied verbatim to the output.
    n_channels : int
        Number of output channels (labelled ch1 … chN).
    channel_gains : dict, optional
        Per-channel gain override.  Keys are channel names (e.g. "ch1").
        Values are either a scalar float (applied uniformly to every entry in
        that channel) or a list of floats with one entry per timeline event.
        Channels not listed use the default gain of 1.
    channel_phases : dict, optional
        Per-channel phase override in radians.  Same shape rules as
        channel_gains.  Channels not listed use the default phase of 0.

    Returns
    -------
    dict
        New JSON dict with "timeline" replaced by
        {"ch1": [...], "ch2": [...], ..., "chN": [...]}.

    Raises
    ------
    ValueError
        If "timeline" is missing, not a list, or a per-channel override list
        has a length that does not match the number of timeline events.
    """
    if "timeline" not in input_json:
        raise ValueError("Input JSON is missing the required \"timeline\" key.")

    source_timeline = input_json["timeline"]
    if not isinstance(source_timeline, list):
        raise ValueError("\"timeline\" must be a JSON array.")

    n_events = len(source_timeline)
    channel_gains = channel_gains or {}
    channel_phases = channel_phases or {}

    def _resolve(overrides: dict, ch: str, default: float) -> List[float]:
        val = overrides.get(ch, default)
        if isinstance(val, list):
            if len(val) != n_events:
                raise ValueError(
                    f"Override list for {ch} has {len(val)} entries but "
                    f"timeline has {n_events} events."
                )
            return val
        return [float(val)] * n_events

    channel_timeline: Dict[str, List[dict]] = {}
    for idx in range(n_channels):
        ch = str(idx)
        gains = _resolve(channel_gains, ch, 1.0)
        phases = _resolve(channel_phases, ch, 0.0)

        entries = []
        for event, g, p in zip(source_timeline, gains, phases):
            entry = copy.deepcopy(event)
            entry["gain"] = g
            entry["phase"] = p
            entries.append(entry)

        channel_timeline[ch] = entries

    output = {k: v for k, v in input_json.items() if k != "timeline"}
    output["timeline"] = channel_timeline
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_channel_map(raw: Optional[str]) -> Dict[str, float]:
    """Parse "0:1.0,1:0.5" into {"ch1": 1.0, "ch2": 0.5}."""
    if not raw:
        return {}
    result: Dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":", 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid channel override \"{token}\"; expected format ch1:1.0"
            )
        ch, val = parts
        try:
            result[ch.strip()] = float(val.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Cannot parse value \"{val}\" for channel \"{ch}\" as float."
            )
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a single-channel pulse sequence JSON to a multi-channel "
            "JSON with independent per-channel timelines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        nargs="?",
        help="Path to the single-channel input JSON file. "
             "Defaults to example_outputs/gre_tr0.json in the repo root.",
    )
    p.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Write JSON output to FILE instead of stdout.",
    )
    p.add_argument(
        "--n-channels",
        type=int, default=8, metavar="N",
        help="Number of output channels.",
    )
    p.add_argument(
        "--gains",
        metavar="0:1.0,1:0.5,...",
        help="Comma-separated channel:gain pairs. Unlisted channels default to 1.",
    )
    p.add_argument(
        "--phases",
        metavar="ch1:0,1:1.5708,...",
        help="Comma-separated channel:phase(rad) pairs. Unlisted channels default to 0.",
    )
    return p


def _default_input() -> Path:
    here = Path(__file__).parent
    candidate = here.parent / "example_outputs" / "gre_tr0.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "No input file specified and could not find example_outputs/gre_tr0.json. "
        "Pass an explicit path as the first argument."
    )


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input) if args.input else _default_input()
    with open(input_path) as fh:
        input_json = json.load(fh)

    channel_gains = _parse_channel_map(args.gains)
    channel_phases = _parse_channel_map(args.phases)

    result = parse_multichannel(
        input_json,
        n_channels=args.n_channels,
        channel_gains=channel_gains,
        channel_phases=channel_phases,
    )

    output_str = json.dumps(result, indent=2, separators=(",", ": "))

    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Multi-channel JSON written to: {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    main()
