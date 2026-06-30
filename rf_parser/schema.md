# RF Parser JSON Schema

Two related formats are produced by this toolchain:

| Format | Produced by | `timeline` type |
|---|---|---|
| **Single-channel** | `rf_waveform_parser.py` (GRE) or the TSE generator | Array of events |
| **Multi-channel** | `multichannel_json_parser.py` | Object keyed `"0"`…`"N-1"` |

Both formats share the same top-level scalar fields and `waveforms` array.

---

## A — Top-level fields

| Field | Type | Description |
|---|---|---|
| `seq_version` | string | Pulseq `.seq` source version, e.g. `"1.5.0"` |
| `tr_index` | int | 0-based TR index (`0` = first repetition) |
| `tr_duration_s` | float | Full TR period in seconds, **including** the tail gap |
| `initial_delay_s` | float | Time from TR start to first RF waveform start (s); equals the duration of the first gap event in `timeline` |
| `tail_delay_s` | float | Time from last RF waveform end to TR end (s); **not** represented as a `timeline` event — stored here instead |
| `n_rf_pulses` | int | Total number of RF pulse events in `timeline` (excitation + refocusing combined) |
| `n_unique_waveforms` | int | Number of entries in the `waveforms` array |
| `n_points` | int | Resampled output length applied to every waveform (e.g. `4096`) |
| `n_bits` | int | DAC bit depth; `max_twos_complement` = 2^(`n_bits`−1) − 1 |

> **Timing identity**: `initial_delay_s` + active RF window + `tail_delay_s` = `tr_duration_s`

---

## B — `waveforms` array

Each entry is one distinct RF pulse shape. Multiple `timeline` pulse events may reference the same `waveform_id` (e.g. all refocusing pulses in TSE share one entry).

| Field | Type | Description |
|---|---|---|
| `waveform_id` | int | 0-based index; referenced by `timeline` pulse events via `waveform_id` |
| `pulse_type` | string | `"excitation"` \| `"refocusing"` \| `"inversion"` |
| `n_points` | int | Length of every sample array in this entry (= top-level `n_points`) |
| `n_bits` | int | DAC bit depth (= top-level `n_bits`) |
| `max_twos_complement` | int | 2^(`n_bits`−1) − 1; the integer that maps to normalised amplitude 1.0 |
| `waveform_raw` | float[n_points] | Magnitude at each sample **in Hz**; peak equals `rf_amplitude_hz` |
| `waveform_normalized` | float[n_points] | Magnitude normalised to [0, 1]; peak = 1.0 |
| `waveform_twos_complement` | int[n_points] | Signed DAC codes; peak maps to `max_twos_complement` |
| `phase_rad` | float[n_points] | Instantaneous phase at each sample in radians |
| `sampling_time_s` | float | Duration of one output sample = `rf_duration_s / n_points` |
| `rf_duration_s` | float | Total RF pulse duration in seconds |
| `rf_amplitude_hz` | float | Peak amplitude in Hz (= maximum of `waveform_raw`) |
| `rf_raster_time_s` | float | Native RF raster interval from the `.seq` file (before resampling) |
| `n_samples_original` | int | Native sample count at `rf_raster_time_s` before resampling to `n_points` |

---

## C — `timeline` events

### Single-channel
`timeline` is a JSON **array**. Events are ordered chronologically and cover the active RF window only (from the start of the first gap to the end of the last pulse). The tail gap is excluded — see `tail_delay_s`.

### Multi-channel
`timeline` is a JSON **object** with string keys `"0"` … `"N-1"` (zero-based channel indices). Each value is an array with the same structure as the single-channel case, plus per-event `gain` and `phase` fields.

---

### Event type: `gap`

A silent interval — no RF is transmitted.

| Field | Type | Present in | Description |
|---|---|---|---|
| `type` | `"gap"` | both | Discriminator |
| `duration_ms` | float | both | Gap duration in milliseconds |
| `gain` | float | multi-channel | Amplitude scale factor (default **1**) |
| `phase` | float | multi-channel | Phase offset in radians (default **0**) |

---

### Event type: `pulse`

An RF transmission event using one of the waveforms in the `waveforms` array.

| Field | Type | Present in | Description |
|---|---|---|---|
| `type` | `"pulse"` | both | Discriminator |
| `waveform_id` | int | both | References `waveforms[].waveform_id` |
| `pulse_type` | string | both | `"excitation"` \| `"refocusing"` — informational, mirrors the waveform entry |
| `duration_ms` | float | both | Pulse duration in ms (= `waveforms[waveform_id].rf_duration_s × 1000`) |
| `gain` | float | multi-channel | Scale applied to `waveform_normalized`; effective peak = `gain × rf_amplitude_hz` |
| `phase` | float | multi-channel | Additional phase (rad) added to `phase_rad` at playback time |

---

## D — Multi-channel `timeline` object

```json
"timeline": {
  "0": [ ...events... ],
  "1": [ ...events... ],
  ...
  "7": [ ...events... ]
}
```

- Keys are **zero-based** channel indices encoded as JSON strings.
- All channels carry the same number of events in the same `type` order; only `gain` and `phase` values differ between channels.
- `gain` and `phase` are **per-event**, not per-channel constants, so different events within one channel can have independent values.

---

## Examples

### Single-channel GRE (3 events)
```json
"timeline": [
  { "type": "gap",   "duration_ms": 0.1 },
  { "type": "pulse", "waveform_id": 0, "pulse_type": "excitation", "duration_ms": 3.0 },
  { "type": "gap",   "duration_ms": 8.9 }
]
```

### Multi-channel GRE, channel 0 vs channel 1 (B1⁺ shim)
```json
"timeline": {
  "0": [
    { "type": "gap",   "duration_ms": 0.1, "gain": 1.0,  "phase": 0.0    },
    { "type": "pulse", "waveform_id": 0, "pulse_type": "excitation", "duration_ms": 3.0, "gain": 1.0,  "phase": 0.0    },
    { "type": "gap",   "duration_ms": 8.9, "gain": 1.0,  "phase": 0.0    }
  ],
  "1": [
    { "type": "gap",   "duration_ms": 0.1, "gain": 0.95, "phase": 0.7854 },
    { "type": "pulse", "waveform_id": 0, "pulse_type": "excitation", "duration_ms": 3.0, "gain": 0.95, "phase": 0.7854 },
    { "type": "gap",   "duration_ms": 8.9, "gain": 0.95, "phase": 0.7854 }
  ]
}
```
