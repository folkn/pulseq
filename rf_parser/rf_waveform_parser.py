#!/usr/bin/env python3
"""
rf_waveform_parser.py  –  CLI entry point for the Pulseq RF waveform parser

Parses a Pulseq .seq file and outputs DAC-ready RF waveform data as JSON.
Multiple RF pulses within a single TR are emitted as separate objects in the
``rf_pulses`` array, each resampled to exactly N points.

Usage
-----
    python rf_waveform_parser.py <seq_file> [options]

Options
-------
    -n / --points  N     Samples per RF pulse          [default: 4096]
    -b / --bits    B     DAC bit depth                 [default: 14]
    -t / --tr      I     TR index to extract (0-based) [default: 0]
    -o / --output  FILE  Write JSON to FILE            [default: stdout]

Output JSON schema
------------------
{
  "seq_version":     "1.5.0",
  "tr_index":        0,
  "tr_duration_s":   0.012,
  "n_rf_pulses":     1,
  "n_points":        4096,
  "n_bits":          14,
  "initial_delay_s": 0.0001,
  "tail_delay_s":    0.0089,
  "rf_pulses": [
    {
      "pulse_index":             0,
      "use":                     "e",
      "n_points":                4096,
      "n_bits":                  14,
      "max_twos_complement":     8191,
      "waveform_raw":            [...],   // Hz, n_points values
      "waveform_normalized":     [...],   // [0,1], n_points values
      "waveform_twos_complement":[...],   // int, n_points values
      "phase_rad":               [...],   // rad, n_points values
      "sampling_time_s":         7.32e-7,
      "rf_duration_s":           0.003,
      "rf_amplitude_hz":         37.22,
      "rf_start_in_tr_s":        0.0001,
      "rf_end_in_tr_s":          0.0031,
      "gap_before_s":            0.0001,
      "gap_after_s":             0.0089,
      "rf_raster_time_s":        1e-6,
      "n_samples_original":      3000
    }
  ]
}

Library API
-----------
    from lib import PulseqParser, WaveformExtractor

    seq = PulseqParser('gre.seq').parse()
    tr  = WaveformExtractor(seq, n_points=4096, n_bits=14).extract_tr(0)
    print(tr)
    print(tr.rf_pulses[0])
"""

import argparse
import sys
from pathlib import Path

# Allow running as  python rf_waveform_parser.py  from any directory
sys.path.insert(0, str(Path(__file__).parent))

from lib import PulseqParser, WaveformExtractor   # noqa: E402  (after sys.path fix)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Extract RF waveform(s) from a Pulseq .seq file and output '
            'DAC-ready JSON (amplitude, phase, timing).'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('seq_file', help='Path to the Pulseq .seq file')
    p.add_argument(
        '-n', '--points', type=int, default=4096, metavar='N',
        help='Samples per RF pulse (same for every pulse in the TR)',
    )
    p.add_argument(
        '-b', '--bits', type=int, default=14, metavar='B',
        help="DAC bit depth for two's-complement encoding",
    )
    p.add_argument(
        '-t', '--tr', type=int, default=0, metavar='I', dest='tr_index',
        help='TR index to extract (0 = first TR)',
    )
    p.add_argument(
        '-o', '--output', metavar='FILE',
        help='Write JSON output to FILE (default: stdout)',
    )
    return p


def _print_summary(tr, file=sys.stderr) -> None:
    sep = '─' * 58
    print(sep, file=file)
    print(f'  RF Waveform Parser  ·  Pulseq v{tr.seq_version}', file=file)
    print(sep, file=file)
    print(f'  TR index            : {tr.tr_index}', file=file)
    print(f'  TR duration         : {tr.tr_duration_s * 1e3:.3f} ms', file=file)
    print(f'  RF pulses in TR     : {tr.n_rf_pulses}', file=file)
    print(f'  Unique waveforms    : {tr.n_unique_waveforms}', file=file)
    print(f'  Samples per pulse   : {tr.n_points}', file=file)
    print(f'  Bit depth           : {tr.n_bits}-bit  '
          f'(max = {tr.waveforms[0].max_twos_complement if tr.waveforms else "N/A"})',
          file=file)
    print(f'  Initial delay       : {tr.initial_delay_s * 1e3:.4f} ms', file=file)
    print(f'  Tail delay          : {tr.tail_delay_s * 1e3:.4f} ms', file=file)
    print(sep, file=file)
    for w in tr.waveforms:
        print(
            f'  Waveform {w.waveform_id:2d}  use={w.use!r:3s}  '
            f'dur={w.rf_duration_s * 1e3:.3f}ms  '
            f'amp={w.rf_amplitude_hz:.4g}Hz  '
            f'native={w.n_samples_original}pts',
            file=file,
        )
    print(sep, file=file)
    # Compact timeline on wrapped lines
    parts = []
    for t in tr.timeline:
        if t['type'] == 'gap':
            parts.append(f"gap({t['duration_ms']:.3f}ms)")
        else:
            parts.append(f"{t['use']}[{t['waveform_id']}]({t['duration_ms']:.2f}ms)")
    print('  Timeline:', file=file)
    line = '    '
    for i, part in enumerate(parts):
        sep_str = ' → ' if i < len(parts) - 1 else ''
        if len(line) + len(part) + len(sep_str) > 72 and line.strip():
            print(line, file=file)
            line = '    '
        line += part + sep_str
    if line.strip():
        print(line, file=file)
    print('─' * 58, file=file)


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    seq = PulseqParser(args.seq_file).parse()
    extractor = WaveformExtractor(seq, n_points=args.points, n_bits=args.bits)
    tr = extractor.extract_tr(args.tr_index)

    _print_summary(tr)

    json_str = tr.to_json(indent=2)
    if args.output:
        Path(args.output).write_text(json_str)
        print(f'JSON written → {args.output}', file=sys.stderr)
    else:
        print(json_str)


if __name__ == '__main__':
    main()
