"""
waveform.py  –  RF waveform extraction, resampling, and DAC encoding

Classes
-------
RFWaveform       : DAC-ready waveform for one unique RF pulse shape
TRWaveform       : deduplicated waveform library + timeline for one TR
WaveformExtractor: builds TRWaveform objects from a SequenceData
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

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
        0.0  →  0
        1.0  →  2^(n_bits-1) - 1   (e.g. 8191 for 14-bit)
    """
    max_val = (1 << (n_bits - 1)) - 1
    return np.round(mag_normalised * max_val).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# RFWaveform
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RFWaveform:
    """
    DAC-ready waveform for one unique RF pulse shape.

    Timing information (start, gaps) lives in the TRWaveform timeline,
    not here.  Two pulses that share the same source shape, amplitude, and
    phase offset reference the same RFWaveform entry.

    Attributes
    ----------
    waveform_id          : index into TRWaveform.waveforms (0-based)
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
    rf_raster_time_s     : native RF sampling interval before resampling (s)
    n_samples_original   : native shape length before resampling
    """

    waveform_id:              int
    use:                      str
    n_points:                 int
    n_bits:                   int
    waveform_raw:             np.ndarray
    waveform_normalized:      np.ndarray
    waveform_twos_complement: np.ndarray
    phase_rad:                np.ndarray
    sampling_time_s:          float
    rf_duration_s:            float
    rf_amplitude_hz:          float
    rf_raster_time_s:         float
    n_samples_original:       int

    @property
    def max_twos_complement(self) -> int:
        return (1 << (self.n_bits - 1)) - 1

    def to_dict(self) -> dict:
        return {
            'waveform_id':              self.waveform_id,
            'use':                      self.use,
            'n_points':                 self.n_points,
            'n_bits':                   self.n_bits,
            'max_twos_complement':      self.max_twos_complement,
            'waveform_raw':             self.waveform_raw.tolist(),
            'waveform_normalized':      self.waveform_normalized.tolist(),
            'waveform_twos_complement': self.waveform_twos_complement.tolist(),
            'phase_rad':                self.phase_rad.tolist(),
            'sampling_time_s':          float(self.sampling_time_s),
            'rf_duration_s':            float(self.rf_duration_s),
            'rf_amplitude_hz':          float(self.rf_amplitude_hz),
            'rf_raster_time_s':         float(self.rf_raster_time_s),
            'n_samples_original':       self.n_samples_original,
        }

    def __repr__(self) -> str:
        return (
            f"RFWaveform(id={self.waveform_id}, use='{self.use}', "
            f"dur={self.rf_duration_s*1e3:.2f}ms, "
            f"amp={self.rf_amplitude_hz:.4g}Hz, "
            f"n_pts={self.n_points}, {self.n_bits}-bit)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRWaveform
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TRWaveform:
    """
    Deduplicated RF waveform library + playback timeline for one TR period.

    Identical RF pulses (same shape, amplitude, phase offset) share a single
    RFWaveform entry in ``waveforms``.  The ``timeline`` list interleaves gap
    and pulse entries to describe the full TR playback order:

        [gap, pulse, gap, pulse, ..., gap]

    Timeline entry schemas
    ----------------------
    Gap   : {"type": "gap",   "duration_ms": <float>}
    Pulse : {"type": "pulse", "waveform_id": <int>, "use": <str>,
             "duration_ms": <float>}

    Attributes
    ----------
    tr_index     : TR number (0-based)
    tr_duration_s: total TR period duration (s)
    seq_version  : e.g. "1.5.0"
    n_points     : samples per waveform (same for every unique waveform)
    n_bits       : DAC bit depth
    waveforms    : list of unique :class:`RFWaveform` objects
    timeline     : ordered list of gap/pulse dicts
    """

    tr_index:      int
    tr_duration_s: float
    seq_version:   str
    n_points:      int
    n_bits:        int
    waveforms:     List[RFWaveform]
    timeline:      List[dict]

    @property
    def n_rf_pulses(self) -> int:
        """Total RF pulses in this TR (counting duplicates)."""
        return sum(1 for t in self.timeline if t['type'] == 'pulse')

    @property
    def n_unique_waveforms(self) -> int:
        return len(self.waveforms)

    @property
    def initial_delay_s(self) -> float:
        """Time from TR start to the first RF pulse (s)."""
        if self.timeline and self.timeline[0]['type'] == 'gap':
            return self.timeline[0]['duration_ms'] * 1e-3
        return 0.0

    @property
    def tail_delay_s(self) -> float:
        """Time from the last RF pulse end to TR end (s)."""
        if self.timeline and self.timeline[-1]['type'] == 'gap':
            return self.timeline[-1]['duration_ms'] * 1e-3
        return 0.0

    def to_dict(self) -> dict:
        return {
            'seq_version':        self.seq_version,
            'tr_index':           self.tr_index,
            'tr_duration_s':      float(self.tr_duration_s),
            'n_rf_pulses':        self.n_rf_pulses,
            'n_unique_waveforms': self.n_unique_waveforms,
            'n_points':           self.n_points,
            'n_bits':             self.n_bits,
            'initial_delay_s':    float(self.initial_delay_s),
            'tail_delay_s':       float(self.tail_delay_s),
            'waveforms':          [w.to_dict() for w in self.waveforms],
            'timeline':           self.timeline,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, separators=(',', ': '))

    def __repr__(self) -> str:
        return (
            f"TRWaveform(tr_index={self.tr_index}, "
            f"tr_dur={self.tr_duration_s*1e3:.1f}ms, "
            f"n_rf={self.n_rf_pulses}, n_unique={self.n_unique_waveforms}, "
            f"n_pts={self.n_points}, {self.n_bits}-bit)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WaveformExtractor
# ─────────────────────────────────────────────────────────────────────────────

class WaveformExtractor:
    """
    Extract and resample RF waveforms from a :class:`SequenceData`.

    Identical pulses (same Pulseq shape IDs, amplitude, and phase offset)
    are stored once in :attr:`TRWaveform.waveforms`.  Playback order and
    gap durations are in :attr:`TRWaveform.timeline`.

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

        Returns
        -------
        TRWaveform with deduplicated waveforms and a gap/pulse timeline.
        """
        tr_start, tr_end, tr_rf_blocks = self._seq.get_tr(tr_index)
        tr_duration = tr_end - tr_start

        waveform_cache: Dict[tuple, int] = {}
        unique_waveforms: List[RFWaveform] = []
        timeline: List[dict] = []

        prev_rf_end = 0.0   # seconds, relative to TR start

        for _pulse_idx, (block_num, rf_id) in enumerate(tr_rf_blocks):
            ev          = self._seq.rf_events[rf_id]
            block_start = self._seq.block_start_times[block_num]
            rf_start    = block_start + ev['delay'] - tr_start
            rf_raster   = self._seq.rf_raster_time

            mag    = self._seq.magnitude_shape(ev['mag_id'])
            n_orig = len(mag)
            phase  = self._seq.phase_shape_rad(ev['phase_id'], n_orig)
            phase  = phase + ev['phase']      # add constant offset (rad)

            rf_duration = n_orig * rf_raster
            rf_end      = rf_start + rf_duration
            gap_before  = rf_start - prev_rf_end
            use         = ev.get('use', 'u')

            # Deduplication key: everything that determines the output arrays
            source_key = (
                ev['mag_id'],
                ev['phase_id'],
                float(ev['amplitude']),
                float(ev['phase']),
                use,
            )

            if source_key not in waveform_cache:
                waveform_id = len(unique_waveforms)
                waveform_cache[source_key] = waveform_id

                t_orig = (np.arange(n_orig, dtype=np.float64) + 0.5) * rf_raster
                t_new  = np.linspace(t_orig[0], t_orig[-1], self._n_points)

                mag_rs        = _resample_linear(mag,   t_orig, t_new)
                phase_rs      = _resample_nearest(phase, t_orig, t_new)
                twos          = _to_twos_complement(mag_rs, self._n_bits)
                waveform_hz   = mag_rs * ev['amplitude']
                sampling_time = rf_duration / self._n_points

                unique_waveforms.append(RFWaveform(
                    waveform_id              = waveform_id,
                    use                      = use,
                    n_points                 = self._n_points,
                    n_bits                   = self._n_bits,
                    waveform_raw             = waveform_hz,
                    waveform_normalized      = mag_rs,
                    waveform_twos_complement = twos,
                    phase_rad                = phase_rs,
                    sampling_time_s          = sampling_time,
                    rf_duration_s            = rf_duration,
                    rf_amplitude_hz          = float(ev['amplitude']),
                    rf_raster_time_s         = rf_raster,
                    n_samples_original       = n_orig,
                ))
            else:
                waveform_id = waveform_cache[source_key]

            # Gap before this pulse, then the pulse itself
            timeline.append({
                'type':        'gap',
                'duration_ms': round(gap_before * 1e3, 6),
            })
            timeline.append({
                'type':        'pulse',
                'waveform_id': waveform_id,
                'use':         use,
                'duration_ms': round(rf_duration * 1e3, 6),
            })

            prev_rf_end = rf_end

        # Tail gap
        timeline.append({
            'type':        'gap',
            'duration_ms': round((tr_duration - prev_rf_end) * 1e3, 6),
        })

        return TRWaveform(
            tr_index      = tr_index,
            tr_duration_s = tr_duration,
            seq_version   = self._seq.version_str,
            n_points      = self._n_points,
            n_bits        = self._n_bits,
            waveforms     = unique_waveforms,
            timeline      = timeline,
        )

    def extract_all_trs(self) -> List[TRWaveform]:
        """Extract waveforms for every TR in the sequence."""
        return [self.extract_tr(i) for i in range(self._seq.n_tr)]
