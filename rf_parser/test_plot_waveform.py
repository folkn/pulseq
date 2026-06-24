#!/usr/bin/env python3
"""
test_plot_waveform.py  –  Visualise the output of rf_waveform_parser.py

Run from inside rf_parser/:
    python test_plot_waveform.py
    python test_plot_waveform.py path/to/your.seq -n 2048 -b 16

Default: uses ../tests/expected_output/write_gre.seq  (bundled GRE test file).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe; switch to 'TkAgg' / 'Qt5Agg'
                               # if you want an interactive window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Import the parser from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from rf_waveform_parser import parse_rf_waveform, parse_seq_file, extract_rf_pulse


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_seq() -> Path:
    """Return the bundled GRE .seq file (copied alongside this script,
    or fall back to the repo's tests/expected_output directory)."""
    here = Path(__file__).parent
    local = here / 'write_gre.seq'
    if local.exists():
        return local
    repo_copy = here.parent / 'tests' / 'expected_output' / 'write_gre.seq'
    if repo_copy.exists():
        return repo_copy
    raise FileNotFoundError(
        "Could not find write_gre.seq. "
        "Pass an explicit path as the first argument."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform(result: dict, seq_file: str, outfile: str = 'rf_waveform.png') -> None:
    """
    Produce a four-panel figure:
      1. Magnitude (Hz) – resampled waveform
      2. Two's-complement integers  (n_bits DAC codes)
      3. Phase (radians)
      4. TR timeline (initial delay, RF pulse, tail delay)

    Also overlays the original-rate waveform (grey) on panel 1 for comparison.
    """
    n_pts   = result['n_points']
    n_bits  = result['n_bits']
    max_val = result['max_twos_complement']
    dt_s    = result['sampling_time_s']
    t_rf    = np.arange(n_pts) * dt_s * 1e3         # ms

    mag_norm  = np.array(result['waveform_normalized'])
    mag_hz    = np.array(result['waveform_raw'])
    twos      = np.array(result['waveform_twos_complement'])
    phase     = np.array(result['phase_rad'])

    rf_dur_ms  = result['rf_duration_s']    * 1e3
    init_ms    = result['initial_delay_s']  * 1e3
    tail_ms    = result['tail_delay_s']     * 1e3
    tr_ms      = result['tr_duration_s']    * 1e3
    amp_hz     = result['rf_amplitude_hz']

    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    fig.suptitle(
        f"RF Waveform  ·  {Path(seq_file).name}  ·  "
        f"{n_pts} pts  ·  {n_bits}-bit two's complement",
        fontsize=13, fontweight='bold',
    )

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)

    # ── Panel 1 : Magnitude in Hz ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_rf, mag_hz, color='steelblue', lw=1.2, label='Resampled')
    ax1.set_xlim(0, rf_dur_ms)
    ax1.set_xlabel('Time within RF pulse (ms)')
    ax1.set_ylabel('Amplitude (Hz)')
    ax1.set_title(f'RF Magnitude  (peak = {amp_hz:.4g} Hz)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='k', lw=0.5)

    # ── Panel 2 : Two's complement ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_rf, twos, color='darkorange', lw=1.2)
    ax2.axhline(max_val, color='red', ls='--', lw=0.8, label=f'Max = {max_val}')
    ax2.set_xlim(0, rf_dur_ms)
    ax2.set_xlabel('Time within RF pulse (ms)')
    ax2.set_ylabel(f'{n_bits}-bit integer')
    ax2.set_title(f"{n_bits}-bit Two's-Complement DAC Codes  (max = {max_val})")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-max_val * 0.05, max_val * 1.1)

    # ── Panel 3 : Phase ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.step(t_rf, phase, color='seagreen', lw=1.2, where='mid')
    ax3.set_xlim(0, rf_dur_ms)
    ax3.set_xlabel('Time within RF pulse (ms)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Phase Waveform')
    # Custom y-ticks at multiples of π/2
    pi_ticks = np.arange(
        np.floor(phase.min() / (np.pi / 2)),
        np.ceil(phase.max()  / (np.pi / 2)) + 1,
    ) * (np.pi / 2)
    pi_labels = []
    for v in pi_ticks:
        n = int(round(v / (np.pi / 2)))
        if n == 0:
            pi_labels.append('0')
        elif n == 2:
            pi_labels.append('π')
        elif n == -2:
            pi_labels.append('-π')
        elif n % 2 == 0:
            pi_labels.append(f'{n//2}π')
        else:
            pi_labels.append(f'{n}π/2')
    ax3.set_yticks(pi_ticks)
    ax3.set_yticklabels(pi_labels)
    ax3.grid(alpha=0.3)

    # ── Panel 4 : TR timeline ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlim(0, tr_ms)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Time in TR (ms)')
    ax4.set_title('TR Timeline')
    ax4.set_yticks([])

    col_init = '#FFC107'   # amber  – initial delay
    col_rf   = '#1565C0'   # blue   – RF pulse
    col_tail = '#43A047'   # green  – tail delay

    # Regions as filled rectangles
    ax4.axvspan(0,                  init_ms,              alpha=0.35, color=col_init)
    ax4.axvspan(init_ms,            init_ms + rf_dur_ms,  alpha=0.45, color=col_rf)
    ax4.axvspan(init_ms + rf_dur_ms, tr_ms,               alpha=0.35, color=col_tail)

    # Brace-style labels
    def mid_label(ax, x0, x1, text, color):
        xm = (x0 + x1) / 2
        ax.text(xm, 0.5, text, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.8, lw=0))

    mid_label(ax4, 0,                  init_ms,              f'Initial delay\n{init_ms:.3f} ms', col_init)
    mid_label(ax4, init_ms,            init_ms + rf_dur_ms,  f'RF pulse\n{rf_dur_ms:.3f} ms',   col_rf)
    mid_label(ax4, init_ms + rf_dur_ms, tr_ms,               f'Tail delay\n{tail_ms:.3f} ms',   col_tail)

    legend_patches = [
        mpatches.Patch(color=col_init, alpha=0.6, label=f'Initial delay  {init_ms:.4g} ms'),
        mpatches.Patch(color=col_rf,   alpha=0.7, label=f'RF pulse       {rf_dur_ms:.4g} ms'),
        mpatches.Patch(color=col_tail, alpha=0.6, label=f'Tail delay     {tail_ms:.4g} ms'),
    ]
    ax4.legend(handles=legend_patches, loc='lower right', fontsize=8)

    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {outfile}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Parameter printout
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(result: dict) -> None:
    print()
    print("=" * 60)
    print("  RF Waveform Parser – Full Parameter Summary")
    print("=" * 60)

    groups = [
        ("Sequence",   ['seq_version', 'n_samples_original', 'rf_raster_time_s']),
        ("Resampled output", ['n_points', 'n_bits', 'max_twos_complement', 'sampling_time_s']),
        ("RF pulse",   ['rf_amplitude_hz', 'rf_duration_s']),
        ("Timing",     ['initial_delay_s', 'tail_delay_s', 'tr_duration_s']),
    ]

    for title, keys in groups:
        print(f"\n  {title}")
        print("  " + "─" * 50)
        for k in keys:
            v = result[k]
            if isinstance(v, float):
                if abs(v) < 0.01 or abs(v) >= 1e4:
                    print(f"    {k:<30s}  {v:.6e}")
                else:
                    print(f"    {k:<30s}  {v:.6f}")
            else:
                print(f"    {k:<30s}  {v}")

    print()
    print("  Waveform arrays (first 6 / last 6 values)")
    print("  " + "─" * 50)
    for arr_key in ['waveform_normalized', 'waveform_twos_complement', 'phase_rad']:
        arr = result[arr_key]
        head = ', '.join(f'{v:.4g}' for v in arr[:6])
        tail = ', '.join(f'{v:.4g}' for v in arr[-6:])
        print(f"    {arr_key}:")
        print(f"      first 6: [{head}]")
        print(f"      last  6: [{tail}]")
    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Test and plot output of rf_waveform_parser.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        'seq_file', nargs='?',
        help='Path to .seq file (default: bundled write_gre.seq)',
    )
    p.add_argument('-n', '--points', type=int, default=4096, metavar='N',
                   help='Number of output samples')
    p.add_argument('-b', '--bits', type=int, default=14, metavar='B',
                   help='DAC bit depth')
    p.add_argument('--rf-index', type=int, default=0, metavar='I',
                   help='Index of RF pulse to extract (0 = first)')
    p.add_argument('-o', '--outfile', default='rf_waveform.png',
                   help='Output image filename')
    p.add_argument('--show', action='store_true',
                   help='Display interactive plot window (requires a display)')
    return p


def main() -> None:
    args = _build_parser().parse_args()

    seq_file = args.seq_file or str(_default_seq())
    print(f"Parsing: {seq_file}")
    print(f"  points={args.points}  bits={args.bits}  rf_index={args.rf_index}")

    result = parse_rf_waveform(
        seq_file=seq_file,
        n_points=args.points,
        n_bits=args.bits,
        rf_index=args.rf_index,
    )

    print_summary(result)
    fig = plot_waveform(result, seq_file, outfile=args.outfile)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
