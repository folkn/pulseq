"""
waveform.py  –  RF waveform extraction, resampling, and DAC encoding

Classes
-------
RFWaveform       : DAC-ready waveform for one RF pulse
TRWaveform       : all RF waveforms for one TR period
WaveformExtractor: builds TRWaveform objects from a SequenceData
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.interpolate import interp1d

from .parser import SequenceData


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample_linear(y: np.ndarray, t_src: np.ndarray,
                     t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation with edge clamping."""
    f = interp1d(t_src, y, kind='linear', bounds_error=False,
                 fill_value=(y[0], y[-1]))
    return np.clip(f(t_dst), 0.0, 1.0)


def _resample_nearest(y: np.ndarray, t_src: np.ndarray,
                      t_dst: np.ndarray) -> np.ndarray:
    """Nearest-neighbour interpolation – preserves discrete phase steps."""
    f = interp1d(t_src, y, kind='nearest', bounds_error=False,
                 fill_value='extrapolate')
    return f(t_dst)


def _to_twos_complement(mag_normalised: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Map a normalised [0, 1] magnitude to N-bit signed two's-complement integers.

    The positive half-range [0, 2^(n_bits-1) - 1] is used so that:
        0.0  →  0
        1.0  →  2^(n_bits-1) - 1   (e.g. 8191 for 14-bit)

    Negative-going waveforms are not expected for magnitude-only output.
    The two's-complement bit-pattern for these positive integers is identical
    to their unsigned binary representation.
    """
    max_val = (1 << (n_bits - 1)) - 1
    return np.round(mag_normalised * max_val).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# RFWaveform
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RFWaveform:
    """
    DAC-ready waveform for a single RF pulse.

    All arrays have exactly ``n_points`` elements and cover the pulse from its
    first sample to its last (i.e. the waveform window == the RF duration).

    Parameters / attributes
    -----------------------
    pulse_index          : position of this pulse within its TR (0-based)
    use                  : Pulseq use type  'e'=excitation, 'r'=refocusing, …
    n_points             : number of output samples
    n_bits               : DAC bit depth
    waveform_raw         : ndarray – amplitude in Hz at each sample
    waveform_normalized  : ndarray – amplitude normalised to [0, 1]
    waveform_twos_complement : ndarray int32 – N-bit two's-complement codes
    phase_rad            : ndarray – instantaneous phase (rad) per sample
    sampling_time_s      : duration of one output sample (s)
    rf_duration_s        : total duration of the RF pulse (s)
    rf_amplitude_hz      : peak RF amplitude (Hz)
    rf_start_in_tr_s     : absolute RF start time within TR (s)
    rf_end_in_tr_s       : absolute RF end time within TR (s)
    gap_before_s         : silent gap from TR start (or prev RF end) to this RF start (s)
    gap_after_s          : silent gap from this RF end to next RF start (or TR end) (s)
    rf_raster_time_s     : native RF sampling interval before resampling (s)
    n_samples_original   : native shape length before resampling
    """

    pulse_index:             int
    use:                     str
    n_points:                int
    n_bits:                  int
    waveform_raw:            np.ndarray
    waveform_normalized:     np.ndarray
    waveform_twos_complement: np.ndarray
    phase_rad:               np.ndarray
    sampling_time_s:         float
    rf_duration_s:           float
    rf_amplitude_hz:         float
    rf_start_in_tr_s:        float
    rf_end_in_tr_s:          float
    gap_before_s:            float
    gap_after_s:             float
    rf_raster_time_s:        float
    n_samples_original:      int

    @property
    def max_twos_complement(self) -> int:
        return (1 << (self.n_bits - 1)) - 1

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            'pulse_index':             self.pulse_index,
            'use':                     self.use,
            'n_points':                self.n_points,
            'n_bits':                  self.n_bits,
            'max_twos_complement':     self.max_twos_complement,
            'waveform_raw':            self.waveform_raw.tolist(),
            'waveform_normalized':     self.waveform_normalized.tolist(),
            'waveform_twos_complement': self.waveform_twos_complement.tolist(),
            'phase_rad':               self.phase_rad.tolist(),
            'sampling_time_s':         float(self.sampling_time_s),
            'rf_duration_s':           float(self.rf_duration_s),
            'rf_amplitude_hz':         float(self.rf_amplitude_hz),
            'rf_start_in_tr_s':        float(self.rf_start_in_tr_s),
            'rf_end_in_tr_s':          float(self.rf_end_in_tr_s),
            'gap_before_s':            float(self.gap_before_s),
            'gap_after_s':             float(self.gap_after_s),
            'rf_raster_time_s':        float(self.rf_raster_time_s),
            'n_samples_original':      self.n_samples_original,
        }

    def __repr__(self) -> str:
        return (
            f"RFWaveform(index={self.pulse_index}, use='{self.use}', "
            f"dur={self.rf_duration_s*1e3:.2f}ms, "
            f"start_in_tr={self.rf_start_in_tr_s*1e3:.3f}ms, "
            f"amp={self.rf_amplitude_hz:.4g}Hz, "
            f"n_pts={self.n_points}, {self.n_bits}-bit)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRWaveform
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TRWaveform:
    """
    Container for all RF waveforms within one TR period.

    Attributes
    ----------
    tr_index    : TR number (0-based)
    tr_duration_s: total TR period duration (s)
    seq_version : e.g. "1.5.0"
    n_points    : samples per waveform (same for every RF pulse)
    n_bits      : DAC bit depth
    rf_pulses   : list of :class:`RFWaveform`, ordered by time within TR
    """

    tr_index:      int
    tr_duration_s: float
    seq_version:   str
    n_points:      int
    n_bits:        int
    rf_pulses:     List[RFWaveform]

    @property
    def n_rf_pulses(self) -> int:
        return len(self.rf_pulses)

    @property
    def initial_delay_s(self) -> float:
        """Time from TR start to the first RF pulse start (s)."""
        return self.rf_pulses[0].gap_before_s if self.rf_pulses else 0.0

    @property
    def tail_delay_s(self) -> float:
        """Time from the last RF pulse end to TR end (s)."""
        return self.rf_pulses[-1].gap_after_s if self.rf_pulses else 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            'seq_version':       self.seq_version,
            'tr_index':          self.tr_index,
            'tr_duration_s':     float(self.tr_duration_s),
            'n_rf_pulses':       self.n_rf_pulses,
            'n_points':          self.n_points,
            'n_bits':            self.n_bits,
            'initial_delay_s':   float(self.initial_delay_s),
            'tail_delay_s':      float(self.tail_delay_s),
            'rf_pulses':         [p.to_dict() for p in self.rf_pulses],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, separators=(',', ': '))

    def __repr__(self) -> str:
        return (
            f"TRWaveform(tr_index={self.tr_index}, "
            f"tr_dur={self.tr_duration_s*1e3:.1f}ms, "
            f"n_rf={self.n_rf_pulses}, n_pts={self.n_points}, "
            f"{self.n_bits}-bit)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WaveformExtractor
# ─────────────────────────────────────────────────────────────────────────────

class WaveformExtractor:
    """
    Extract and resample RF waveforms from a :class:`SequenceData`.

    Each RF pulse within a TR is resampled independently to exactly
    ``n_points`` samples spanning its native duration (start → end of
    the RF waveform itself, not the enclosing TR or block).

    Parameters
    ----------
    seq      : parsed sequence
    n_points : number of output samples per RF pulse  (default 4096)
    n_bits   : DAC bit depth for two's-complement encoding  (default 14)
    """

    def __init__(
        self,
        seq: SequenceData,
        n_points: int = 4096,
        n_bits: int = 14,
    ) -> None:
        if n_points < 1:
            raise ValueError(f"n_points must be ≥ 1, got {n_points}.")
        if n_bits < 2 or n_bits > 32:
            raise ValueError(f"n_bits must be in [2, 32], got {n_bits}.")
        self._seq      = seq
        self._n_points = n_points
        self._n_bits   = n_bits

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_tr(self, tr_index: int = 0) -> TRWaveform:
        """
        Extract all RF waveforms for TR number ``tr_index``.

        Parameters
        ----------
        tr_index : int
            0-based TR index.  Use :attr:`SequenceData.n_tr` to know the range.

        Returns
        -------
        TRWaveform
        """
        tr_start, tr_end, tr_rf_blocks = self._seq.get_tr(tr_index)
        tr_duration = tr_end - tr_start

        rf_waveforms: List[RFWaveform] = []

        for pulse_idx, (block_num, rf_id) in enumerate(tr_rf_blocks):
            ev          = self._seq.rf_events[rf_id]
            block_start = self._seq.block_start_times[block_num]
            rf_start    = block_start + ev['delay'] - tr_start  # relative to TR start
            rf_raster   = self._seq.rf_raster_time

            # ── Native magnitude & phase shapes ──────────────────────────────
            mag = self._seq.magnitude_shape(ev['mag_id'])
            n_orig = len(mag)
            phase = self._seq.phase_shape_rad(ev['phase_id'], n_orig)
            phase = phase + ev['phase']               # add constant offset (rad)

            rf_duration = n_orig * rf_raster
            rf_end      = rf_start + rf_duration

            # ── Silent gap before and after this pulse ───────────────────────
            if pulse_idx == 0:
                gap_before = rf_start                 # TR start → this RF start
            else:
                prev_end = rf_waveforms[-1].rf_end_in_tr_s
                gap_before = rf_start - prev_end

            if pulse_idx == len(tr_rf_blocks) - 1:
                gap_after = tr_duration - rf_end      # this RF end → TR end
            else:
                next_block, next_rf_id = tr_rf_blocks[pulse_idx + 1]
                next_ev    = self._seq.rf_events[next_rf_id]
                next_start = (
                    self._seq.block_start_times[next_block]
                    + next_ev['delay']
                    - tr_start
                )
                gap_after = next_start - rf_end

            # ── Resample to n_points ─────────────────────────────────────────
            t_orig = (np.arange(n_orig, dtype=np.float64) + 0.5) * rf_raster
            t_new  = np.linspace(t_orig[0], t_orig[-1], self._n_points)

            mag_rs   = _resample_linear(mag,   t_orig, t_new)
            phase_rs = _resample_nearest(phase, t_orig, t_new)

            twos = _to_twos_complement(mag_rs, self._n_bits)
            waveform_hz = mag_rs * ev['amplitude']

            sampling_time = rf_duration / self._n_points

            rf_waveforms.append(RFWaveform(
                pulse_index              = pulse_idx,
                use                      = ev.get('use', 'u'),
                n_points                 = self._n_points,
                n_bits                   = self._n_bits,
                waveform_raw             = waveform_hz,
                waveform_normalized      = mag_rs,
                waveform_twos_complement = twos,
                phase_rad                = phase_rs,
                sampling_time_s          = sampling_time,
                rf_duration_s            = rf_duration,
                rf_amplitude_hz          = float(ev['amplitude']),
                rf_start_in_tr_s         = rf_start,
                rf_end_in_tr_s           = rf_end,
                gap_before_s             = gap_before,
                gap_after_s              = gap_after,
                rf_raster_time_s         = rf_raster,
                n_samples_original       = n_orig,
            ))

        return TRWaveform(
            tr_index      = tr_index,
            tr_duration_s = tr_duration,
            seq_version   = self._seq.version_str,
            n_points      = self._n_points,
            n_bits        = self._n_bits,
            rf_pulses     = rf_waveforms,
        )

    def extract_all_trs(self) -> List[TRWaveform]:
        """Extract waveforms for every TR in the sequence."""
        return [self.extract_tr(i) for i in range(self._seq.n_tr)]
