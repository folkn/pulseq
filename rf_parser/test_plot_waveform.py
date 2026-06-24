#!/usr/bin/env python3
"""
test_plot_waveform.py  –  Test and visualise rf_waveform_parser output

Produces one PNG figure per .seq file tested:
  • One column per unique waveform shape (deduplicated)
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
# Palette
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
}


def _pulse_color(use: str) -> str:
    return _USE_COLORS.get(use, '#37474F')


# ─────────────────────────────────────────────────────────────────────────────
# Per-waveform three-panel figure (magnitude, twos-comp, phase)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_waveform_panels(axes, wf, title_prefix: str = '') -> None:
    """Fill three axes (mag, codes, phase) for one RFWaveform."""
    t_ms = np.arange(wf.n_points) * wf.sampling_time_s * 1e3

    ax_mag, ax_code, ax_ph = axes
    col = _pulse_color(wf.use)

    # Magnitude
    ax_mag.plot(t_ms, wf.waveform_normalized, color=col, lw=1.1)
    ax_mag.set_ylim(-0.05, 1.15)
    ax_mag.set_ylabel('Norm. amp.')
    ax_mag.set_title(
        f"{title_prefix}Waveform {wf.waveform_id}  "
        f"use='{wf.use}'  {wf.rf_amplitude_hz:.4g} Hz  "
        f"{wf.rf_duration_s * 1e3:.2f} ms",
        fontsize=9,
    )
    ax_mag.axhline(0, color='k', lw=0.4)
    ax_mag.grid(alpha=0.25)

    # Two's complement
    max_v = wf.max_twos_complement
    ax_code.plot(t_ms, wf.waveform_twos_complement, color=col, lw=1.1)
    ax_code.axhline(max_v, color='#B71C1C', ls='--', lw=0.7,
                    label=f'max={max_v}')
    ax_code.set_ylabel(f'{wf.n_bits}-bit code')
    ax_code.set_ylim(-max_v * 0.05, max_v * 1.15)
    ax_code.legend(fontsize=7, loc='upper right')
    ax_code.grid(alpha=0.25)

    # Phase
    ax_ph.step(t_ms, wf.phase_rad, where='mid', color=col, lw=1.1)
    ax_ph.set_xlabel('Time in pulse (ms)')
    ax_ph.set_ylabel('Phase (rad)')
    ax_ph.grid(alpha=0.25)
    pmin = np.floor(wf.phase_rad.min() / (np.pi / 2)) * (np.pi / 2)
    pmax = np.ceil(wf.phase_rad.max()  / (np.pi / 2)) * (np.pi / 2)
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
# TR timeline panel  (reconstructed from the gap/pulse timeline list)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_timeline(ax, tr: TRWaveform) -> None:
    tr_ms = tr.tr_duration_s * 1e3
    ax.set_xlim(0, tr_ms)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time in TR (ms)')
    ax.set_title('TR Timeline', fontsize=9)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.2)

    legend_seen = set()
    patches = []
    t_ms = 0.0

    for entry in tr.timeline:
        if entry['type'] == 'gap':
            t_ms += entry['duration_ms']
        else:
            dur = entry['duration_ms']
            wid = entry['waveform_id']
            col = _pulse_color(entry['use'])
            ax.axvspan(t_ms, t_ms + dur, alpha=0.55, color=col)
            xm = t_ms + dur / 2
            ax.text(xm, 0.5,
                    f"W{wid}\n{entry['use']}\n{dur:.1f}ms",
                    ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold',
                    clip_on=True)
            if wid not in legend_seen:
                patches.append(mpatches.Patch(
                    color=col, alpha=0.7,
                    label=f"W{wid} ({entry['use']})  {dur:.2f}ms",
                ))
                legend_seen.add(wid)
            t_ms += dur

    # Shade initial and tail gaps
    init_ms = tr.initial_delay_s * 1e3
    tail_start_ms = tr_ms - tr.tail_delay_s * 1e3
    if init_ms > 0:
        ax.axvspan(0, init_ms, alpha=0.20, color=_GAP_COLORS['before'],
                   label=f'init {init_ms:.3f}ms')
    if tail_start_ms < tr_ms:
        ax.axvspan(tail_start_ms, tr_ms, alpha=0.20, color=_GAP_COLORS['after'],
                   label=f'tail {tr.tail_delay_s*1e3:.3f}ms')

    ax.legend(handles=patches, loc='upper right', fontsize=7,
              framealpha=0.8, ncol=min(4, len(patches)))


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def plot_tr(tr: TRWaveform, seq_file: str, outfile: str = 'rf_waveform.png') -> str:
    """
    Build and save a figure for the given TRWaveform.

    Columns = unique waveforms (deduplicated).
    Rows 0-2: per-waveform panels  |  Row 3: shared TR timeline.
    Returns the saved file path.
    """
    n_wf = tr.n_unique_waveforms
    if n_wf == 0:
        print('  No RF pulses – nothing to plot.')
        return ''

    col_width  = max(3.5, min(6.0, 20 / n_wf))
    fig_width  = col_width * n_wf
    fig_height = 10 + (1 if n_wf > 3 else 0)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.suptitle(
        f'{Path(seq_file).name}  ·  TR {tr.tr_index}  '
        f'({tr.tr_duration_s*1e3:.1f} ms)  ·  '
        f'{tr.n_rf_pulses} pulse(s)  ·  '
        f'{tr.n_unique_waveforms} unique  ·  '
        f'{tr.n_points} pts / {tr.n_bits}-bit',
        fontsize=11, fontweight='bold',
    )

    gs = gridspec.GridSpec(
        4, n_wf, figure=fig,
        height_ratios=[2.0, 1.8, 1.8, 1.4],
    )

    for col, wf in enumerate(tr.waveforms):
        axes = [fig.add_subplot(gs[row, col]) for row in range(3)]
        _plot_waveform_panels(axes, wf)
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
          f'|  {tr.n_rf_pulses} pulse(s)  |  {tr.n_unique_waveforms} unique waveform(s)')
    print(sep)
    print(f'  TR duration   : {tr.tr_duration_s * 1e3:.3f} ms')
    print(f'  Initial delay : {tr.initial_delay_s * 1e6:.1f} µs '
          f'({tr.initial_delay_s * 1e3:.4f} ms)')
    print(f'  Tail delay    : {tr.tail_delay_s * 1e3:.4f} ms')
    print()
    for wf in tr.waveforms:
        print(f'  ── Waveform {wf.waveform_id} ────────────────────────────────────────')
        print(f'    use          : {wf.use!r}')
        print(f'    duration     : {wf.rf_duration_s * 1e3:.3f} ms')
        print(f'    amplitude    : {wf.rf_amplitude_hz:.6g} Hz')
        print(f'    native pts   : {wf.n_samples_original}')
        print(f'    sampling dt  : {wf.sampling_time_s * 1e9:.2f} ns')
        print(f'    max DAC code : {wf.max_twos_complement}')
        wn = wf.waveform_normalized
        wt = wf.waveform_twos_complement
        ph = wf.phase_rad
        head = ', '.join(f'{v:.4g}' for v in wn[:4])
        tail_v = ', '.join(f'{v:.4g}' for v in wn[-4:])
        print(f'    norm (first4): [{head}]')
        print(f'    norm (last4) : [{tail_v}]')
        head_t = ', '.join(str(v) for v in wt[:4])
        print(f'    codes(first4): [{head_t}]')
        head_ph = ', '.join(f'{v:.4g}' for v in ph[:4])
        print(f'    phase(first4): [{head_ph}] rad')
    print()
    print('  Timeline:')
    parts = []
    for t in tr.timeline:
        if t['type'] == 'gap':
            parts.append(f"gap({t['duration_ms']:.3f}ms)")
        else:
            parts.append(f"{t['use']}[W{t['waveform_id']}]({t['duration_ms']:.2f}ms)")
    line = '    '
    for i, part in enumerate(parts):
        sep_str = ' → ' if i < len(parts) - 1 else ''
        if len(line) + len(part) + len(sep_str) > 70 and line.strip():
            print(line)
            line = '    '
        line += part + sep_str
    if line.strip():
        print(line)
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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
