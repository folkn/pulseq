#!/usr/bin/env python3
"""
rf_waveform_parser.py  –  Standalone Pulseq .seq → RF-waveform extractor for DAC output

Parses a Pulseq .seq file (v1.2.x – v1.5.x), finds the first (or selected) RF pulse
across one TR period, resamples the magnitude and phase to exactly N points, and emits
all parameters needed to drive an arbitrary-waveform DAC.

No dependency on the pypulseq package – only numpy, scipy, and the standard library.

Usage
-----
    python rf_waveform_parser.py <seq_file> [options]

Options
-------
    -n / --points  N       Total output samples         [default: 4096]
    -b / --bits    B       DAC bit depth                [default: 14]
    -o / --output  FILE    Write JSON to FILE           [default: stdout]
    --rf-index     I       RF pulse to extract (0-based)[default: 0]

Output fields (JSON)
--------------------
    waveform_raw              float list – magnitude in Hz at each output sample
    waveform_normalized       float list – magnitude normalised to [0, 1]
    waveform_twos_complement  int list   – N-bit signed two's-complement integers,
                                           peak maps to 2^(N-1)-1  (always ≥ 0 for
                                           magnitude-only output)
    phase_rad                 float list – instantaneous phase at each sample (rad)
    sampling_time_s           float      – duration of one output sample (s)
    rf_duration_s             float      – total RF pulse duration (s)
    initial_delay_s           float      – TR frame start → RF pulse start (s)
    tail_delay_s              float      – RF pulse end → TR frame end (s)
    tr_duration_s             float      – one complete TR period (s)
    rf_amplitude_hz           float      – peak RF amplitude (Hz)
    n_points                  int        – number of output samples
    n_bits                    int        – bit depth
    max_twos_complement       int        – 2^(n_bits-1) - 1  (e.g. 8191 for 14-bit)
    rf_raster_time_s          float      – native RF sampling interval (s)
    n_samples_original        int        – native shape length before resampling
    seq_version               str        – e.g. "1.5.0"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Shape decompression  (mirrors pypulseq/decompress_shape.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _decompress_shape(num_samples: int, data: np.ndarray) -> np.ndarray:
    """
    Decompress a Pulseq run-length-encoded shape.

    Storage convention
    ------------------
    * If len(data) == num_samples: stored verbatim – return a copy.
    * Otherwise: data encodes first-differences with run-length packing.
      Identical consecutive differences are stored as (v, v, count-2).
      Reconstruct by decoding the RLE stream then applying cumsum.
    """
    data = np.asarray(data, dtype=np.float64)
    pack_len = len(data)

    if pack_len == num_samples:
        return data.copy()

    out = np.empty(num_samples, dtype=np.float64)
    diffs = data[1:] - data[:-1]
    markers = np.flatnonzero(diffs == 0.0)

    cp = 0   # pointer into compressed data
    up = 0   # pointer into uncompressed output

    for m in markers:
        skip = m - cp
        if skip < 0:          # false positive already consumed
            continue
        if skip > 0:          # copy verbatim block before the run
            out[up : up + skip] = data[cp : cp + skip]
            cp += skip
            up += skip
        # packed run: [v, v, count-2]  →  count = data[cp+2] + 2 repeats of v
        rep = int(data[cp + 2] + 2)
        out[up : up + rep] = data[cp]
        cp += 3
        up += rep

    remaining = pack_len - cp
    if remaining > 0:
        out[up : up + remaining] = data[cp:]

    return np.cumsum(out)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level .seq file reader
# ─────────────────────────────────────────────────────────────────────────────

def _split_sections(path: str) -> Dict[str, List[str]]:
    """
    Read a .seq file and return a dict mapping section names to their
    non-empty, non-comment lines.
    """
    sections: Dict[str, List[str]] = {}
    current = 'HEADER'
    buf: List[str] = []

    with open(path, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                sections[current] = buf
                current = line[1:-1]
                buf = []
            else:
                buf.append(line)

    sections[current] = buf
    return sections


def _parse_version(lines: List[str]) -> Tuple[int, int, int]:
    d: Dict[str, int] = {}
    for line in lines:
        parts = line.split()
        # take only the leading digit to handle e.g. "0post1"
        d[parts[0]] = int(parts[1][0])
    return d.get('major', 0), d.get('minor', 0), d.get('revision', 0)


def _parse_definitions(lines: List[str]) -> dict:
    defs: dict = {}
    for line in lines:
        parts = line.split()
        key = parts[0]
        vals = parts[1:]
        try:
            nums = [float(v) for v in vals]
            defs[key] = nums[0] if len(nums) == 1 else np.array(nums)
        except ValueError:
            defs[key] = ' '.join(vals)
    return defs


def _parse_blocks(
    lines: List[str], block_raster: float, version_combined: int
) -> Tuple[dict, dict]:
    """
    Parse [BLOCKS] section.

    Returns
    -------
    blocks : dict  block_num → list of event IDs [rf, gx, gy, gz, adc, ext]
    block_durations : dict  block_num → duration in seconds
    """
    blocks: dict = {}
    block_durations: dict = {}
    for line in lines:
        vals = list(map(int, line.split()))
        num = vals[0]
        dur_counts = vals[1]
        event_ids = vals[2:]        # [rf, gx, gy, gz, adc, ext]
        blocks[num] = event_ids
        block_durations[num] = dur_counts * block_raster
    return blocks, block_durations


def _parse_rf_events(lines: List[str], version_combined: int) -> dict:
    """
    Parse [RF] section.

    Returns dict  rf_id → {
        amplitude      : Hz
        mag_id         : shape library index for magnitude
        phase_id       : shape library index for phase  (0 = no phase shape)
        time_shape_id  : shape library index for time axis (0 = uniform)
        center         : s  (centre of pulse within the delay window)
        delay          : s  (delay before pulse starts within the block)
        freq_ppm       : ppm
        phase_ppm      : rad / MHz
        freq           : Hz
        phase          : rad  (constant phase offset)
        use            : str  e/r/i/s/p/o/u
    }
    """
    events: dict = {}
    for line in lines:
        parts = line.split()
        rf_id = int(float(parts[0]))

        if version_combined >= 1_005_000:
            # id ampl mag_id phase_id time_shape_id center delay
            #    freq_ppm phase_ppm freq phase use
            ev = {
                'amplitude':     float(parts[1]),
                'mag_id':        int(float(parts[2])),
                'phase_id':      int(float(parts[3])),
                'time_shape_id': int(float(parts[4])),
                'center':        float(parts[5]) * 1e-6,   # µs → s
                'delay':         float(parts[6]) * 1e-6,   # µs → s
                'freq_ppm':      float(parts[7]),
                'phase_ppm':     float(parts[8]),
                'freq':          float(parts[9]),
                'phase':         float(parts[10]),          # rad
                'use':           parts[11] if len(parts) > 11 else 'u',
            }
        elif version_combined >= 1_004_000:
            # id ampl mag_id phase_id time_shape_id delay freq phase
            ev = {
                'amplitude':     float(parts[1]),
                'mag_id':        int(float(parts[2])),
                'phase_id':      int(float(parts[3])),
                'time_shape_id': int(float(parts[4])),
                'center':        0.0,
                'delay':         float(parts[5]) * 1e-6,
                'freq_ppm':      0.0,
                'phase_ppm':     0.0,
                'freq':          float(parts[6]),
                'phase':         float(parts[7]),
                'use':           'u',
            }
        else:
            # id ampl mag_id phase_id delay freq phase
            ev = {
                'amplitude':     float(parts[1]),
                'mag_id':        int(float(parts[2])),
                'phase_id':      int(float(parts[3])),
                'time_shape_id': 0,
                'center':        0.0,
                'delay':         float(parts[4]) * 1e-6,
                'freq_ppm':      0.0,
                'phase_ppm':     0.0,
                'freq':          float(parts[5]),
                'phase':         float(parts[6]),
                'use':           'u',
            }
        events[rf_id] = ev
    return events


def _parse_shapes(lines: List[str]) -> Dict[int, np.ndarray]:
    """
    Parse [SHAPES] section.

    Returns dict  shape_id → decompressed ndarray.
    Each decompressed array has num_samples elements in whatever units the
    shape encodes (normalised [0,1] for magnitude; normalised cycles [0,1]
    for phase, where 1.0 == 2π rad).
    """
    shape_lib: Dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('shape_id'):
            shape_id = int(line.split()[1])
            i += 1
            if i >= len(lines) or not lines[i].startswith('num_samples'):
                break
            num_samples = int(lines[i].split()[1])
            i += 1
            raw: List[float] = []
            while i < len(lines) and not lines[i].startswith('shape_id'):
                raw.append(float(lines[i]))
                i += 1
            data = np.array(raw, dtype=np.float64)
            shape_lib[shape_id] = _decompress_shape(num_samples, data)
        else:
            i += 1
    return shape_lib


# ─────────────────────────────────────────────────────────────────────────────
# High-level sequence parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_seq_file(path: str) -> dict:
    """
    Parse a Pulseq .seq file (v1.2.x – v1.5.x) into a structured dict.

    Returns
    -------
    dict with keys:
        version           : (major, minor, revision)
        version_combined  : int
        definitions       : dict
        rf_raster_time    : s
        grad_raster_time  : s
        block_raster_time : s
        blocks            : {block_num: [rf_id, gx_id, gy_id, gz_id, adc_id, ext_id]}
        block_durations   : {block_num: duration_s}
        rf_events         : {rf_id: event_dict}
        shape_library     : {shape_id: ndarray}
    """
    sections = _split_sections(path)

    version_major, version_minor, version_revision = _parse_version(
        sections.get('VERSION', [])
    )
    version_combined = (
        version_major * 1_000_000
        + version_minor * 1_000
        + version_revision
    )

    defs = _parse_definitions(sections.get('DEFINITIONS', []))
    rf_raster_time    = float(defs.get('RadiofrequencyRasterTime', 1e-6))
    grad_raster_time  = float(defs.get('GradientRasterTime', 10e-6))
    block_raster_time = float(defs.get('BlockDurationRaster', 10e-6))

    blocks, block_durations = _parse_blocks(
        sections.get('BLOCKS', []), block_raster_time, version_combined
    )
    rf_events  = _parse_rf_events(sections.get('RF', []), version_combined)
    shape_lib  = _parse_shapes(sections.get('SHAPES', []))

    return {
        'version':          (version_major, version_minor, version_revision),
        'version_combined': version_combined,
        'definitions':      defs,
        'rf_raster_time':   rf_raster_time,
        'grad_raster_time': grad_raster_time,
        'block_raster_time': block_raster_time,
        'blocks':           blocks,
        'block_durations':  block_durations,
        'rf_events':        rf_events,
        'shape_library':    shape_lib,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RF pulse extraction
# ─────────────────────────────────────────────────────────────────────────────

def _block_start_times(seq: dict) -> Dict[int, float]:
    """Return cumulative start time in seconds for every block."""
    times: Dict[int, float] = {}
    t = 0.0
    for bn in sorted(seq['blocks'].keys()):
        times[bn] = t
        t += seq['block_durations'][bn]
    return times


def extract_rf_pulse(seq: dict, rf_index: int = 0) -> dict:
    """
    Locate the rf_index-th RF pulse, recover its native waveform, and compute
    all timing relationships relative to the enclosing TR period.

    Returns
    -------
    dict with keys:
        mag_shape      : ndarray [0, 1]  – normalised magnitude, n_orig samples
        phase_shape    : ndarray (rad)   – per-sample phase, n_orig samples
        t_axis         : ndarray (s)     – sample mid-points within the RF pulse
        amplitude_hz   : float           – peak amplitude (Hz)
        rf_raster_time : float (s)
        rf_duration    : float (s)
        n_orig         : int
        initial_delay_s: float (s)  – TR frame start → RF pulse start
        tail_delay_s   : float (s)  – RF pulse end   → TR frame end
        tr_duration_s  : float (s)
        rf_event       : dict  (raw parsed RF event)
        block_num      : int   (block containing this RF event)
    """
    blocks        = seq['blocks']
    block_dur     = seq['block_durations']
    rf_events     = seq['rf_events']
    shape_lib     = seq['shape_library']
    rf_raster     = seq['rf_raster_time']

    start_times = _block_start_times(seq)
    total_dur   = sum(block_dur.values())

    # Collect all RF-containing blocks in order
    rf_blocks = [
        (bn, events[0])
        for bn in sorted(blocks.keys())
        for events in [blocks[bn]]
        if events[0] != 0
    ]

    if not rf_blocks:
        raise ValueError("No RF events found in the sequence.")
    if rf_index >= len(rf_blocks):
        raise ValueError(
            f"rf_index={rf_index} is out of range; "
            f"sequence has {len(rf_blocks)} RF pulse(s)."
        )

    target_block, target_rf_id = rf_blocks[rf_index]
    target_rf  = rf_events[target_rf_id]
    target_use = target_rf.get('use', 'u')
    target_mag = target_rf['mag_id']

    # TR start = t = 0 (beginning of block 1)
    tr_start = 0.0

    # TR end = start of the next block with the same mag_id and use type
    # (identifies the start of the subsequent TR repetition)
    tr_end = total_dur
    for bn, rf_id in rf_blocks[rf_index + 1:]:
        ev = rf_events[rf_id]
        if (ev['mag_id'] == target_mag
                and ev.get('use', 'u') == target_use):
            tr_end = start_times[bn]
            break

    tr_duration = tr_end - tr_start

    # Absolute time at which the RF waveform begins playing
    rf_start_abs = start_times[target_block] + target_rf['delay']
    initial_delay_s = rf_start_abs - tr_start   # TR frame start → RF start

    # ── Magnitude shape ──────────────────────────────────────────────────────
    mag_id = target_rf['mag_id']
    if mag_id not in shape_lib:
        raise ValueError(
            f"Magnitude shape id={mag_id} is missing from [SHAPES] section."
        )
    mag_shape = shape_lib[mag_id].copy()
    mag_shape = np.clip(mag_shape, 0.0, None)           # must be non-negative
    peak = mag_shape.max()
    if peak == 0.0:
        raise ValueError("Magnitude shape is all zeros.")
    mag_shape /= peak                                    # normalise to [0, 1]

    n_orig      = len(mag_shape)
    rf_duration = n_orig * rf_raster

    # ── Phase shape ───────────────────────────────────────────────────────────
    phase_id = target_rf['phase_id']
    if phase_id != 0 and phase_id in shape_lib:
        # Pulseq stores phase as normalised cycles [0, 1] → convert to radians
        phase_shape = shape_lib[phase_id] * (2.0 * np.pi)

        # If phase and magnitude shapes have different lengths, resample phase
        if len(phase_shape) != n_orig:
            t_p = np.linspace(0.0, 1.0, len(phase_shape))
            t_m = np.linspace(0.0, 1.0, n_orig)
            phase_shape = interp1d(
                t_p, phase_shape, kind='nearest', fill_value='extrapolate'
            )(t_m)
    else:
        phase_shape = np.zeros(n_orig, dtype=np.float64)

    # Add the constant per-event phase offset (rad)
    phase_shape = phase_shape + target_rf['phase']

    # ── Time axis (midpoint of each RF raster interval) ───────────────────────
    t_axis = (np.arange(n_orig, dtype=np.float64) + 0.5) * rf_raster

    tail_delay_s = tr_duration - initial_delay_s - rf_duration

    return {
        'mag_shape':       mag_shape,
        'phase_shape':     phase_shape,
        't_axis':          t_axis,
        'amplitude_hz':    float(target_rf['amplitude']),
        'rf_raster_time':  rf_raster,
        'rf_duration':     rf_duration,
        'n_orig':          n_orig,
        'initial_delay_s': initial_delay_s,
        'tail_delay_s':    tail_delay_s,
        'tr_duration_s':   tr_duration,
        'rf_event':        target_rf,
        'block_num':       target_block,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def _resample(
    mag_shape: np.ndarray,
    phase_shape: np.ndarray,
    t_axis: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample magnitude and phase to exactly n_points uniformly within the
    same time span as t_axis.

    Uses linear interpolation for magnitude (smooth continuous signal) and
    nearest-neighbour for phase (piecewise-constant, avoids erroneous
    intermediate values at 0/π transitions).

    Returns (t_new, mag_resampled, phase_resampled).
    """
    t_new = np.linspace(t_axis[0], t_axis[-1], n_points)

    f_mag = interp1d(
        t_axis, mag_shape,
        kind='linear',
        bounds_error=False,
        fill_value=(mag_shape[0], mag_shape[-1]),
    )
    mag_resampled = np.clip(f_mag(t_new), 0.0, 1.0)

    f_ph = interp1d(
        t_axis, phase_shape,
        kind='nearest',
        bounds_error=False,
        fill_value='extrapolate',
    )
    phase_resampled = f_ph(t_new)

    return t_new, mag_resampled, phase_resampled


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_rf_waveform(
    seq_file: str,
    n_points: int = 4096,
    n_bits: int = 14,
    rf_index: int = 0,
) -> dict:
    """
    Parse a Pulseq .seq file and return a dict of DAC-ready waveform data.

    Parameters
    ----------
    seq_file  : path to the .seq file
    n_points  : number of output samples  (the total length of the waveform)
    n_bits    : DAC bit depth
    rf_index  : which RF pulse to extract (0 = first)

    Returns
    -------
    dict  (all JSON-serialisable) with the fields documented in the module
    docstring.
    """
    seq   = parse_seq_file(seq_file)
    pulse = extract_rf_pulse(seq, rf_index=rf_index)

    t_new, mag_rs, phase_rs = _resample(
        pulse['mag_shape'],
        pulse['phase_shape'],
        pulse['t_axis'],
        n_points,
    )

    # Two's-complement encoding
    # Peak of normalised magnitude (1.0) maps to 2^(n_bits-1) - 1
    max_val: int = (1 << (n_bits - 1)) - 1
    twos_comp = np.round(mag_rs * max_val).astype(np.int32)

    # Waveform in physical units (Hz)
    waveform_hz = mag_rs * pulse['amplitude_hz']

    sampling_time_s = pulse['rf_duration'] / n_points

    v_maj, v_min, v_rev = seq['version']

    return {
        # ── Waveform ─────────────────────────────────────────────────────────
        'waveform_raw':             waveform_hz.tolist(),
        'waveform_normalized':      mag_rs.tolist(),
        'waveform_twos_complement': twos_comp.tolist(),
        'phase_rad':                phase_rs.tolist(),
        # ── Timing ───────────────────────────────────────────────────────────
        'sampling_time_s':          float(sampling_time_s),
        'rf_duration_s':            float(pulse['rf_duration']),
        'initial_delay_s':          float(pulse['initial_delay_s']),
        'tail_delay_s':             float(pulse['tail_delay_s']),
        'tr_duration_s':            float(pulse['tr_duration_s']),
        # ── Amplitude ────────────────────────────────────────────────────────
        'rf_amplitude_hz':          float(pulse['amplitude_hz']),
        # ── Configuration ────────────────────────────────────────────────────
        'n_points':                 int(n_points),
        'n_bits':                   int(n_bits),
        'max_twos_complement':      int(max_val),
        # ── Native RF parameters ─────────────────────────────────────────────
        'rf_raster_time_s':         float(pulse['rf_raster_time']),
        'n_samples_original':       int(pulse['n_orig']),
        'seq_version':              f"{v_maj}.{v_min}.{v_rev}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract the RF waveform from a Pulseq .seq file and output "
            "DAC-ready amplitude, phase, and timing parameters as JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('seq_file', help='Path to the Pulseq .seq file')
    p.add_argument(
        '-n', '--points',
        type=int, default=4096, metavar='N',
        help='Number of output samples (total waveform length)',
    )
    p.add_argument(
        '-b', '--bits',
        type=int, default=14, metavar='B',
        help='DAC bit depth (signed two\'s-complement)',
    )
    p.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Write JSON output to FILE instead of stdout',
    )
    p.add_argument(
        '--rf-index',
        type=int, default=0, metavar='I',
        help='Index of the RF pulse to extract (0 = first)',
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    result = parse_rf_waveform(
        seq_file=args.seq_file,
        n_points=args.points,
        n_bits=args.bits,
        rf_index=args.rf_index,
    )

    # Pretty-print scalar fields to stderr so they're visible even when
    # redirecting stdout to a file
    summary_keys = [
        'seq_version', 'n_samples_original', 'n_points', 'n_bits',
        'rf_raster_time_s', 'sampling_time_s', 'rf_duration_s',
        'initial_delay_s', 'tail_delay_s', 'tr_duration_s',
        'rf_amplitude_hz', 'max_twos_complement',
    ]
    print("─" * 54, file=sys.stderr)
    print("RF Waveform Parser – Summary", file=sys.stderr)
    print("─" * 54, file=sys.stderr)
    for k in summary_keys:
        v = result[k]
        if isinstance(v, float):
            print(f"  {k:<28s}  {v:.6g}", file=sys.stderr)
        else:
            print(f"  {k:<28s}  {v}", file=sys.stderr)
    print("─" * 54, file=sys.stderr)

    output = json.dumps(result, indent=2, separators=(',', ': '))
    if args.output:
        Path(args.output).write_text(output)
        print(f"Waveform written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
