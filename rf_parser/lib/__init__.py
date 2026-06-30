"""
pulseq_rf_parser  –  Standalone Pulseq .seq → DAC waveform library

Public API:
    PulseqParser    : parse a .seq file into SequenceData
    SequenceData    : structured representation of the parsed sequence
    WaveformExtractor: extract and resample RF waveforms
    TRWaveform      : all RF waveforms for one TR period
    RFWaveform      : DAC-ready waveform for a single RF pulse
"""

from .parser import PulseqParser, SequenceData
from .waveform import WaveformExtractor, TRWaveform, RFWaveform

__all__ = [
    'PulseqParser',
    'SequenceData',
    'WaveformExtractor',
    'TRWaveform',
    'RFWaveform',
]
