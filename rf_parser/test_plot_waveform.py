#!/usr/bin/env python3
"""
test_plot_waveform.py  –  Test and visualise rf_waveform_parser output

Produces one PNG figure per .seq file tested:
  • One column of panels per RF pulse in the TR
  • Row 0: normalised magnitude waveform
  • Row 1: two's-complement DAC codes
  • Row 2: phase (rad)
  • Row 3 (shared): TR timeline showing all pulses and gaps

Run from rf_parser/:
    python test_plot_waveform.py                          # GRE only (default)
    python test_plot_waveform.py --all                    # GRE + TSE
    python test_plot_waveform.py sequences/write_tse.seq  # explicit file
    python test_plot_waveform.py --all -n 2048 -b 12     # custom config
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from lib import PulseqParser, WaveformExtractor, TRWaveform


# ─────────────────────────────────────────────────────────────────────────────
# Palette (cycles for many pulses)
# ─────────────────────────────────────────────────────────────────────────────
_USE_COLORS = {
    'e': '#1565C0',   # excitation – blue
    'r': '#C62828',   # refocusing – red
    'i': '#6A1B9A',   # inversion  – purple
    's': '#E65100',   # saturation – orange
    'u': '#37474F',   # undefined  – grey
}
_GAP_COLORS = {
    'before': '#FFA726',   # amber
    'after':  '#66BB6A',   # green
    'inter':  '#EF9A9A',   # light red
}


def _pulse_color(use: str) -> str:
    return _USE_COLORS.get(use, '#37474F')


# ─────────────────────────────────────────────────────────────────────────────
# Per-pulse three-panel figure (magnitude, twos-comp, phase)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pulse_panels(axes, pulse, title_prefix: str = '') -> None:
    """Fill three axes (mag, codes, phase) for one RFWaveform."""
    t_ms = np.arange(pulse.n_points) * pulse.sampling_time_s * 1e3

    ax_mag, ax_code, ax_ph = axes
    col = _pulse_color(pulse.use)

    # Magnitude
    ax_mag.plot(t_ms, pulse.waveform_normalized, color=col, lw=1.1)
    ax_mag.set_ylim(-0.05, 1.15)
    ax_mag.set_ylabel('Norm. amp.')
    ax_mag.set_title(
        f"{title_prefix}Pulse {pulse.pulse_index}  "
        f"use='{pulse.use}'  {pulse.rf_amplitude_hz:.4g} Hz  "
        f"{pulse.rf_duration_s * 1e3:.2f} ms",
        fontsize=9,
    )
    ax_mag.axhline(0, color='k', lw=0.4)
    ax_mag.grid(alpha=0.25)

    # Two's complement
    max_v = pulse.max_twos_complement
    ax_code.plot(t_ms, pulse.waveform_twos_complement, color=col, lw=1.1)
    ax_code.axhline(max_v, color='#B71C1C', ls='--', lw=0.7,
                    label=f'max={max_v}')
    ax_code.set_ylabel(f'{pulse.n_bits}-bit code')
    ax_code.set_ylim(-max_v * 0.05, max_v * 1.15)
    ax_code.legend(fontsize=7, loc='upper right')
    ax_code.grid(alpha=0.25)

    # Phase
    ax_ph.step(t_ms, pulse.phase_rad, where='mid', color=col, lw=1.1)
    ax_ph.set_xlabel('Time in pulse (ms)')
    ax_ph.set_ylabel('Phase (rad)')
    ax_ph.grid(alpha=0.25)
    # π-tick labels
    pmin = np.floor(pulse.phase_rad.min() / (np.pi / 2)) * (np.pi / 2)
    pmax = np.ceil(pulse.phase_rad.max()  / (np.pi / 2)) * (np.pi / 2)
    ticks = np.arange(pmin, pmax + 0.01, np.pi / 2)
    if len(ticks) <= 8:
        def _label(v):
            n = int(round(v / (np.pi / 2)))
            if n == 0:  return '0'
            if n == 2:  return 'π'
            if n == -2: return '-π'
            return f'{n}π/2' if n % 2 else f'{n//2}π'
        ax_ph.set_yticks(ticks)
        ax_ph.set_yticklabels([_label(v) for v in ticks])


# ─────────────────────────────────────────────────────────────────────────────
# TR timeline panel
# ─────────────────────────────────────────────────────────────────────────────

def _plot_timeline(ax, tr: TRWaveform) -> None:
    tr_ms = tr.tr_duration_s * 1e3
    ax.set_xlim(0, tr_ms)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time in TR (ms)')
    ax.set_title('TR Timeline', fontsize=9)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.2)

    patches = []
    for i, p in enumerate(tr.rf_pulses):
        s_ms = p.rf_start_in_tr_s * 1e3
        e_ms = p.rf_end_in_tr_s   * 1e3
        col  = _pulse_color(p.use)
        ax.axvspan(s_ms, e_ms, alpha=0.55, color=col)
        xm = (s_ms + e_ms) / 2
        label = f"P{i}\n{p.use}\n{(e_ms-s_ms):.1f}ms"
        ax.text(xm, 0.5, label, ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold',
                clip_on=True)
        patches.append(mpatches.Patch(
            color=col, alpha=0.7,
            label=f'P{i} ({p.use})  {p.rf_duration_s*1e3:.2f}ms  '
                  f'@ {p.rf_start_in_tr_s*1e3:.2f}ms',
        ))

    # Shade initial delay and tail delay
    init_ms = tr.initial_delay_s * 1e3
    tail_start_ms = tr.rf_pulses[-1].rf_end_in_tr_s * 1e3 if tr.rf_pulses else 0
    if init_ms > 0:
        ax.axvspan(0, init_ms, alpha=0.20, color=_GAP_COLORS['before'],
                   label=f'init delay {init_ms:.3f}ms')
    if tail_start_ms < tr_ms:
        ax.axvspan(tail_start_ms, tr_ms, alpha=0.20, color=_GAP_COLORS['after'],
                   label=f'tail delay {tr.tail_delay_s*1e3:.3f}ms')

    ax.legend(handles=patches, loc='upper right', fontsize=7,
              framealpha=0.8, ncol=min(4, len(patches)))


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def plot_tr(tr: TRWaveform, seq_file: str, outfile: str = 'rf_waveform.png') -> str:
    """
    Build and save a figure for the given TRWaveform.

    Layout:
        Rows 0-2: per-pulse panels (mag, codes, phase)  ← one column each pulse
        Row  3  : shared TR timeline
    Returns the saved file path.
    """
    n_pulses = tr.n_rf_pulses
    if n_pulses == 0:
        print('  No RF pulses – nothing to plot.')
        return ''

    n_rows = 4          # mag / codes / phase / timeline
    n_cols = n_pulses

    col_width  = max(3.5, min(6.0, 20 / n_cols))
    fig_width  = col_width * n_cols
    fig_height = 10 + (1 if n_pulses > 3 else 0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.suptitle(
        f'{Path(seq_file).name}  ·  TR {tr.tr_index}  '
        f'({tr.tr_duration_s*1e3:.1f} ms)  ·  '
        f'{tr.n_rf_pulses} RF pulse(s)  ·  '
        f'{tr.n_points} pts / {tr.n_bits}-bit',
        fontsize=11, fontweight='bold',
    )

    # Grid: top 3 rows = per-pulse panels, bottom row spans all columns
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        height_ratios=[2.0, 1.8, 1.8, 1.4],
    )

    for col, pulse in enumerate(tr.rf_pulses):
        axes = [fig.add_subplot(gs[row, col]) for row in range(3)]
        _plot_pulse_panels(axes, pulse)
        # Only label y-axis on left column to save space
        if col > 0:
            for ax in axes:
                ax.set_ylabel('')

    ax_tl = fig.add_subplot(gs[3, :])
    _plot_timeline(ax_tl, tr)

    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'  Figure saved → {outfile}')
    return outfile


# ─────────────────────────────────────────────────────────────────────────────
# Summary printout
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(tr: TRWaveform) -> None:
    sep = '=' * 62
    print(sep)
    print(f'  TR {tr.tr_index}  |  Pulseq v{tr.seq_version}  '
          f'|  {tr.n_rf_pulses} RF pulse(s)')
    print(sep)
    print(f'  TR duration   : {tr.tr_duration_s * 1e3:.3f} ms')
    print(f'  Initial delay : {tr.initial_delay_s * 1e6:.1f} µs '
          f'({tr.initial_delay_s * 1e3:.4f} ms)')
    print(f'  Tail delay    : {tr.tail_delay_s * 1e3:.4f} ms')
    print()
    for p in tr.rf_pulses:
        print(f'  ── Pulse {p.pulse_index} ──────────────────────────────────────')
        print(f'    use          : {p.use!r}')
        print(f'    duration     : {p.rf_duration_s * 1e3:.3f} ms')
        print(f'    amplitude    : {p.rf_amplitude_hz:.6g} Hz')
        print(f'    start in TR  : {p.rf_start_in_tr_s * 1e3:.4f} ms')
        print(f'    end in TR    : {p.rf_end_in_tr_s * 1e3:.4f} ms')
        print(f'    gap before   : {p.gap_before_s * 1e3:.4f} ms')
        print(f'    gap after    : {p.gap_after_s * 1e3:.4f} ms')
        print(f'    native pts   : {p.n_samples_original}')
        print(f'    sampling dt  : {p.sampling_time_s * 1e9:.2f} ns')
        print(f'    max DAC code : {p.max_twos_complement}')
        # Show first/last few waveform values
        wn = p.waveform_normalized
        wt = p.waveform_twos_complement
        ph = p.phase_rad
        head = ', '.join(f'{v:.4g}' for v in wn[:4])
        tail = ', '.join(f'{v:.4g}' for v in wn[-4:])
        print(f'    norm (first4): [{head}]')
        print(f'    norm (last4) : [{tail}]')
        head_t = ', '.join(str(v) for v in wt[:4])
        print(f'    codes(first4): [{head_t}]')
        head_ph = ', '.join(f'{v:.4g}' for v in ph[:4])
        print(f'    phase(first4): [{head_ph}] rad')
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _default_seq_files() -> list:
    here = Path(__file__).parent
    candidates = [
        here / 'sequences' / 'write_gre.seq',
        here / 'sequences' / 'write_tse.seq',
        here.parent / 'tests' / 'expected_output' / 'write_gre.seq',
    ]
    return [str(p) for p in candidates if p.exists()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Test and visualise rf_waveform_parser output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        'seq_files', nargs='*',
        help='Path(s) to .seq file(s). Default: sequences/write_gre.seq',
    )
    p.add_argument('--all', action='store_true',
                   help='Test all .seq files in sequences/')
    p.add_argument('-n', '--points', type=int, default=4096, metavar='N',
                   help='Samples per RF pulse')
    p.add_argument('-b', '--bits',   type=int, default=14,   metavar='B',
                   help='DAC bit depth')
    p.add_argument('-t', '--tr',     type=int, default=0,    metavar='I',
                   dest='tr_index', help='TR index to extract')
    p.add_argument('--show', action='store_true',
                   help='Open interactive plot window (needs display)')
    return p


def main() -> None:
    args = _build_parser().parse_args()

    seq_dir = Path(__file__).parent / 'sequences'
    if args.all:
        files = sorted(str(p) for p in seq_dir.glob('*.seq'))
        if not files:
            print('No .seq files found in sequences/  – copy some there first.')
            sys.exit(1)
    elif args.seq_files:
        files = args.seq_files
    else:
        gre = seq_dir / 'write_gre.seq'
        fallback = Path(__file__).parent.parent / 'tests' / 'expected_output' / 'write_gre.seq'
        if gre.exists():
            files = [str(gre)]
        elif fallback.exists():
            files = [str(fallback)]
        else:
            print('Cannot find a default .seq file.  Pass one as an argument.')
            sys.exit(1)

    for seq_file in files:
        print(f'\n{"=" * 62}')
        print(f'  Parsing: {seq_file}')
        print(f'  points={args.points}  bits={args.bits}  tr_index={args.tr_index}')
        print(f'{"=" * 62}')

        seq = PulseqParser(seq_file).parse()
        print(f'  Sequence: v{seq.version_str}  TRs={seq.n_tr}')
        extractor = WaveformExtractor(seq, n_points=args.points, n_bits=args.bits)
        tr = extractor.extract_tr(args.tr_index)

        print_summary(tr)

        stem = Path(seq_file).stem
        outfile = f'{stem}_tr{args.tr_index}.png'
        plot_tr(tr, seq_file, outfile=outfile)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
