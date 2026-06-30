#!/usr/bin/env python3
"""Generate a hard (rectangular block) pulse GRE sequence example."""
import sys
import importlib.metadata

# Patch missing package metadata (dev install without egg-info)
_real_version = importlib.metadata.version
def _patched_version(name):
    if name == 'pypulseq': return '1.5.0'
    return _real_version(name)
importlib.metadata.version = _patched_version

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
import pypulseq as pp

system = pp.Opts(
    max_grad=28, grad_unit='mT/m',
    max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
)

seq = pp.Sequence(system=system)

# Hard (block) pulse: 500 µs, 30° flip angle, excitation
flip_deg = 30.0
rf = pp.make_block_pulse(
    flip_angle=flip_deg * np.pi / 180,
    duration=500e-6,
    system=system,
    use='excitation',
)

# TR = 10 ms: RF block + delay to fill TR
tr = 10e-3
post_delay = pp.make_delay(
    tr - rf.delay - rf.t[-1] - system.rf_ringdown_time
)

for _ in range(4):
    seq.add_block(rf)
    seq.add_block(post_delay)

out = Path(__file__).parent / 'write_hard_pulse.seq'
seq.write(str(out))
print(f'Written  : {out}')
print(f'Flip     : {flip_deg}°')
print(f'Amplitude: {abs(rf.signal.max()):.4f} Hz')
print(f'Duration : {rf.t[-1]*1e6:.0f} µs  ({len(rf.signal)} samples)')
print(f'TR       : {tr*1e3:.0f} ms  |  TRs=4')
