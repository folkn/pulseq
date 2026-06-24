"""
parser.py  –  Pulseq .seq file parser

Classes
-------
PulseqParser   : reads a .seq file and returns a SequenceData
SequenceData   : immutable structured view of a parsed Pulseq sequence

Supports Pulseq file format v1.2.x through v1.5.x.
No dependency on the pypulseq package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal: shape decompression
# ─────────────────────────────────────────────────────────────────────────────

def _decompress_shape(num_samples: int, data: np.ndarray) -> np.ndarray:
    """
    Decompress a Pulseq run-length-encoded shape (mirrors decompress_shape.py).

    If len(data) == num_samples the shape is stored verbatim; return a copy.
    Otherwise decode the RLE-on-differences stream and apply cumsum.

    Encoding: repeated consecutive first-differences are stored as
    (v, v, count - 2) triplets; single-occurrence diffs are stored as-is.
    """
    data = np.asarray(data, dtype=np.float64)
    pack_len = len(data)

    if pack_len == num_samples:
        return data.copy()

    out = np.empty(num_samples, dtype=np.float64)
    diffs = data[1:] - data[:-1]
    markers = np.flatnonzero(diffs == 0.0)

    cp = 0   # compressed pointer
    up = 0   # uncompressed pointer

    for m in markers:
        skip = m - cp
        if skip < 0:       # already past this marker (false positive)
            continue
        if skip > 0:       # verbatim block preceding the run
            out[up : up + skip] = data[cp : cp + skip]
            cp += skip
            up += skip
        rep = int(data[cp + 2] + 2)
        out[up : up + rep] = data[cp]
        cp += 3
        up += rep

    remaining = pack_len - cp
    if remaining > 0:
        out[up : up + remaining] = data[cp:]

    return np.cumsum(out)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: section-level parsers
# ─────────────────────────────────────────────────────────────────────────────

def _split_sections(path: str) -> Dict[str, List[str]]:
    """Return {section_name: [non-blank, non-comment lines]} for every section."""
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
        d[parts[0]] = int(parts[1][0])   # first char handles e.g. "0post1"
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
    lines: List[str], block_raster: float
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Parse [BLOCKS] section.

    Returns
    -------
    blocks         : {block_num: [rf_id, gx_id, gy_id, gz_id, adc_id, ext_id]}
    block_durations: {block_num: duration_s}
    """
    blocks: Dict[int, List[int]] = {}
    durations: Dict[int, float] = {}
    for line in lines:
        vals = list(map(int, line.split()))
        num = vals[0]
        blocks[num] = vals[2:]                   # event IDs: rf, gx, gy, gz, adc, ext
        durations[num] = vals[1] * block_raster
    return blocks, durations


def _parse_rf_events(lines: List[str], version_combined: int) -> Dict[int, dict]:
    """
    Parse [RF] section.

    Returns {rf_id: { amplitude, mag_id, phase_id, time_shape_id, center,
                      delay, freq_ppm, phase_ppm, freq, phase, use }}
    All times are in seconds, amplitude in Hz, phase in radians.
    """
    events: Dict[int, dict] = {}
    for line in lines:
        parts = line.split()
        rf_id = int(float(parts[0]))

        if version_combined >= 1_005_000:
            # id ampl mag_id phase_id time_shape_id center(µs) delay(µs)
            #    freq_ppm phase_ppm freq phase use
            ev: dict = {
                'amplitude':     float(parts[1]),
                'mag_id':        int(float(parts[2])),
                'phase_id':      int(float(parts[3])),
                'time_shape_id': int(float(parts[4])),
                'center':        float(parts[5]) * 1e-6,
                'delay':         float(parts[6]) * 1e-6,
                'freq_ppm':      float(parts[7]),
                'phase_ppm':     float(parts[8]),
                'freq':          float(parts[9]),
                'phase':         float(parts[10]),
                'use':           parts[11] if len(parts) > 11 else 'u',
            }
        elif version_combined >= 1_004_000:
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

    Returns {shape_id: decompressed_ndarray}.
    Magnitude shapes are in normalised [0, 1]; phase shapes are in normalised
    cycles [0, 1] (multiply by 2π to get radians).
    """
    shape_lib: Dict[int, np.ndarray] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith('shape_id'):
            i += 1
            continue
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
        shape_lib[shape_id] = _decompress_shape(num_samples, np.array(raw))
    return shape_lib


# ─────────────────────────────────────────────────────────────────────────────
# Public data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SequenceData:
    """
    Immutable container for a parsed Pulseq .seq file.

    Attributes
    ----------
    version          : (major, minor, revision)
    version_combined : int  e.g. 1_005_000
    definitions      : raw [DEFINITIONS] dict
    rf_raster_time   : s
    grad_raster_time : s
    block_raster_time: s
    blocks           : {block_num: [rf_id, gx_id, gy_id, gz_id, adc_id, ext_id]}
    block_durations  : {block_num: duration_s}
    rf_events        : {rf_id: event_dict}
    shape_library    : {shape_id: decompressed_ndarray}
    source_path      : str (file that was parsed)
    """
    version:          Tuple[int, int, int]
    version_combined: int
    definitions:      dict
    rf_raster_time:   float
    grad_raster_time: float
    block_raster_time: float
    blocks:           Dict[int, List[int]]
    block_durations:  Dict[int, float]
    rf_events:        Dict[int, dict]
    shape_library:    Dict[int, np.ndarray]
    source_path:      str = ''

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def version_str(self) -> str:
        v = self.version
        return f"{v[0]}.{v[1]}.{v[2]}"

    @cached_property
    def block_start_times(self) -> Dict[int, float]:
        """Cumulative start time (s) for each block."""
        times: Dict[int, float] = {}
        t = 0.0
        for bn in sorted(self.blocks.keys()):
            times[bn] = t
            t += self.block_durations[bn]
        return times

    @cached_property
    def total_duration(self) -> float:
        """Total sequence duration (s)."""
        return sum(self.block_durations.values())

    @cached_property
    def rf_block_sequence(self) -> List[Tuple[int, int, float]]:
        """
        Ordered list of (block_num, rf_id, block_start_s) for every block
        that contains an RF event, in temporal order.
        """
        return [
            (bn, self.blocks[bn][0], self.block_start_times[bn])
            for bn in sorted(self.blocks.keys())
            if self.blocks[bn][0] != 0
        ]

    @cached_property
    def n_tr(self) -> int:
        """Number of detected TR periods."""
        return len(self._tr_boundaries)

    @cached_property
    def _tr_boundaries(self) -> List[Tuple[float, float, List[Tuple[int, int]]]]:
        """
        Detect TR boundaries.

        The TR 'anchor' is the (mag_id, use) of the chronologically first RF
        pulse.  Each consecutive pair of anchor occurrences delimits one TR.
        All RF pulses whose block start time falls within [tr_start, tr_end)
        are assigned to that TR.

        Returns
        -------
        list of (tr_start_s, tr_end_s, [(block_num, rf_id), ...])
        """
        rf_seq = self.rf_block_sequence
        if not rf_seq:
            return []

        # Identify anchor: first RF's (mag_id, use)
        _, first_rf_id, _ = rf_seq[0]
        first_ev = self.rf_events[first_rf_id]
        anchor_key = (first_ev['mag_id'], first_ev.get('use', 'u'))

        anchors = [
            (bn, rf_id, bt)
            for bn, rf_id, bt in rf_seq
            if (self.rf_events[rf_id]['mag_id'],
                self.rf_events[rf_id].get('use', 'u')) == anchor_key
        ]

        trs: List[Tuple[float, float, List[Tuple[int, int]]]] = []
        for i, (a_bn, a_rf_id, a_t) in enumerate(anchors):
            tr_start = 0.0 if i == 0 else a_t
            tr_end   = anchors[i + 1][2] if i + 1 < len(anchors) else self.total_duration

            tr_rf = [
                (bn, rf_id)
                for bn, rf_id, bt in rf_seq
                if tr_start <= bt < tr_end
            ]
            trs.append((tr_start, tr_end, tr_rf))

        return trs

    def get_tr(self, tr_index: int = 0
               ) -> Tuple[float, float, List[Tuple[int, int]]]:
        """
        Return (tr_start_s, tr_end_s, [(block_num, rf_id), ...])
        for TR number tr_index (0-based).
        """
        bnd = self._tr_boundaries
        if not bnd:
            raise ValueError("No RF events found – cannot determine TR.")
        if tr_index < 0 or tr_index >= len(bnd):
            raise IndexError(
                f"tr_index={tr_index} out of range; "
                f"sequence has {len(bnd)} TR period(s)."
            )
        return bnd[tr_index]

    def magnitude_shape(self, mag_id: int) -> np.ndarray:
        """Return normalised [0, 1] magnitude shape, raising if missing."""
        if mag_id not in self.shape_library:
            raise KeyError(f"Magnitude shape id={mag_id} not in [SHAPES].")
        raw = self.shape_library[mag_id].copy()
        raw = np.clip(raw, 0.0, None)
        peak = raw.max()
        if peak == 0.0:
            raise ValueError(f"Magnitude shape id={mag_id} is all zeros.")
        return raw / peak

    def phase_shape_rad(self, phase_id: int, n_mag: int) -> np.ndarray:
        """
        Return phase shape in radians (n_mag samples).

        phase_id == 0  →  all-zero phase array of length n_mag.
        Otherwise decompressed shape values (cycles ×2π) interpolated to
        n_mag samples if they differ in length.
        """
        if phase_id == 0 or phase_id not in self.shape_library:
            return np.zeros(n_mag, dtype=np.float64)

        raw = self.shape_library[phase_id] * (2.0 * np.pi)   # cycles → rad
        if len(raw) == n_mag:
            return raw

        # Lengths differ: nearest-neighbour resample to n_mag
        idx = np.round(np.linspace(0, len(raw) - 1, n_mag)).astype(int)
        return raw[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public parser
# ─────────────────────────────────────────────────────────────────────────────

class PulseqParser:
    """
    Parse a Pulseq .seq file into a :class:`SequenceData` object.

    Parameters
    ----------
    path : str or Path
        Path to the .seq file.

    Examples
    --------
    >>> seq = PulseqParser('gre.seq').parse()
    >>> print(seq.version_str, seq.n_tr)
    """

    def __init__(self, path: str) -> None:
        self._path = str(path)

    def parse(self) -> SequenceData:
        """Read and parse the file, returning a :class:`SequenceData`."""
        sections = _split_sections(self._path)

        version_major, version_minor, version_revision = _parse_version(
            sections.get('VERSION', [])
        )
        version_combined = (
            version_major * 1_000_000
            + version_minor * 1_000
            + version_revision
        )

        defs = _parse_definitions(sections.get('DEFINITIONS', []))
        rf_raster    = float(defs.get('RadiofrequencyRasterTime', 1e-6))
        grad_raster  = float(defs.get('GradientRasterTime', 10e-6))
        block_raster = float(defs.get('BlockDurationRaster', 10e-6))

        blocks, block_durations = _parse_blocks(
            sections.get('BLOCKS', []), block_raster
        )
        rf_events  = _parse_rf_events(sections.get('RF', []), version_combined)
        shape_lib  = _parse_shapes(sections.get('SHAPES', []))

        return SequenceData(
            version          = (version_major, version_minor, version_revision),
            version_combined = version_combined,
            definitions      = defs,
            rf_raster_time   = rf_raster,
            grad_raster_time = grad_raster,
            block_raster_time= block_raster,
            blocks           = blocks,
            block_durations  = block_durations,
            rf_events        = rf_events,
            shape_library    = shape_lib,
            source_path      = self._path,
        )
