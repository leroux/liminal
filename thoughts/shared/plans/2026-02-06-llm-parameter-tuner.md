# LLM-Assisted Parameter Tuner — Implementation Plan

## Overview

Add an AI prompt box to the Parameters page of both pedals (Reverb FDN + Lossy). The user types a free-form description, Claude generates a parameter JSON via the `claude-agent-sdk`, params auto-apply to sliders, and auto-play triggers. Multi-turn conversation so the user can iteratively refine.

## Current State Analysis

Both pedals have identical parameter architecture:
- `engine/params.py` defines `default_params()` dict and `PARAM_RANGES`
- GUI has `_read_params_from_ui()` and `_write_params_to_ui()` for round-tripping
- Preset loading already does exactly what we need: JSON dict → `_write_params_to_ui()`
- Both GUIs use tkinter with a notebook (tabs). Parameters tab is the first tab.

### Key Discoveries:
- Reverb params page: two-column layout inside a scroll canvas (`gui/gui.py:114-358`)
- Lossy params page: single-column scrollable layout (`lossy/gui/gui.py:123-253`)
- Both GUIs already run renders in background threads (`threading.Thread`)
- `pyproject.toml` has `requires-python = ">=3.12"`, dependencies managed via `uv`
- Lossy has `CHOICE_RANGES` dict for integer/enum params (`lossy/engine/params.py:157-168`) — need to handle these in validation

## Desired End State

Both GUIs have an AI prompt box at the bottom of the Parameters page. User types "dark cathedral reverb" or "glitchy internet radio", hits Enter, Claude responds with parameter JSON, sliders auto-update, audio auto-plays. Multi-turn: "darker", "more modulation", "undo that". Works with user's existing Claude Code authentication.

### Verification:
1. Launch reverb GUI, type a prompt, see sliders move and hear the result
2. Launch lossy GUI, same test
3. Multi-turn: send follow-up prompt, confirm Claude adjusts incrementally
4. "New Session" resets context, next prompt starts fresh
5. "Undo" reverts to pre-LLM params

## What We're NOT Doing

- No separate AI tab — prompt lives on the Parameters page
- No model picker — hardcoded Opus 4.6
- No fallback JSON parsing — rely on structured output
- No offline/error recovery — just show the error
- No streaming display of response — wait for full response then apply
- No preset auto-save after LLM generation (user can use existing Save Preset)

## Implementation Approach

Shared `LLMTuner` class handles all Claude SDK interaction. Each GUI instantiates it with its own parameter schema and guide text. The prompt box is a simple tkinter widget group (text input + buttons + response display) embedded at the bottom of the existing Parameters page in both GUIs.

---

## Phase 1: Shared Guide Text Module

### Overview
Create a single file with condensed parameter descriptions for both pedals. Both GUIs import from here for the system prompt. No duplication.

### Changes Required:

#### 1. New file: `gui/llm_guide_text.py`

**File**: `gui/llm_guide_text.py` (new)

```python
"""Condensed parameter guides for the LLM tuner system prompt.

Both pedals import from here — single source of truth.
"""

REVERB_GUIDE = """You are an expert audio DSP engineer tuning an 8-node Feedback Delay Network reverb.

SIGNAL CHAIN: Input -> Pre-delay -> Input Diffusion (allpass chain) -> FDN Loop -> Wet/Dry Mix -> Output
FDN Loop: Read 8 delay lines -> one-pole damping -> feedback matrix multiply -> scale by feedback gain + saturate (tanh) -> write back to delay lines -> sum weighted outputs

PARAMETERS AND RANGES:

Global:
- feedback_gain (0.0-2.0): Energy recirculation. 0=no reverb, 0.85=medium room, 0.95+=long tail. >1.0 WILL explode unless saturation is turned up.
- wet_dry (0.0-1.0): 0=dry only, 0.5=equal blend, 1.0=100% reverb.
- diffusion (0.0-0.7): Allpass chain smearing. 0=sharp attacks, 0.5+=heavy smearing. 4 stages internally.
- saturation (0.0-1.0): Tanh soft-clipping in feedback loop. 0=clean/linear, 0.3=warm, 0.7+=aggressive distortion. Prevents explosion when feedback>1.0.
- stereo_width (0.0-1.0): 0=mono, 1=full stereo spread.
- pre_delay (0-11025 samples): Silence before reverb. 0-441=intimate, 882-2205=medium room, 3528+=large hall. (44100 samples = 1 second)

Matrix:
- matrix_type: "householder" (smooth, default), "hadamard", "diagonal" (metallic/comb), "random_orthogonal", "circulant" (ring), "stautner_puckette" (classic paired)

Per-node (8 values each):
- delay_times (1-13230 samples): Delay lengths. Short <660=small resonant space, 2205-3528=medium room, long=hall. Use prime-ish ratios for density.
- damping_coeffs (0.0-0.99): One-pole lowpass per node. 0=bright, 0.3=warm, 0.7+=dark/muffled.
- input_gains (0.0-0.5): How much input feeds each node. Default 0.125 (equal).
- output_gains (0.0-2.0): Each node's contribution to output. 0=silent, 1=normal, >1=amplified.
- node_pans (-1.0 to 1.0): Stereo position per node. -1=left, 0=center, 1=right.

Modulation:
- mod_master_rate (0.0-1000.0 Hz): LFO speed. 0=off, 0.1=slow evolve, 2=chorus, 80+=FM territory.
- mod_depth_delay (0.0-100.0 samples per node): Delay time swing. 3-5=subtle chorus, 20+=pitch wobble.
- mod_depth_damping (0.0-0.5 per node): Brightness modulation. Creates breathing bright/dark.
- mod_depth_output (0.0-1.0 per node): Output gain modulation. Tremolo-like amplitude variation.
- mod_depth_matrix (0.0-1.0): Blend toward second matrix over time.
- mod_correlation (0.0-1.0): Phase spread. 1=all in sync, 0=maximum decorrelation (wider stereo).
- mod_waveform: 0=sine, 1=triangle, 2=sample_and_hold (stepped random).
- mod_node_rate_mult (0.25-4.0 per node): Per-node LFO rate = master * multiplier. Use integer ratios for rhythmic relationships.

RECIPES:
- Natural room: feedback 0.7-0.9, damping 0.2-0.4, diffusion 0.4-0.5, householder, delays 20-80ms
- Infinite drone: feedback 1.0-1.5, saturation 0.3-0.6, low damping
- Metallic/comb: diagonal matrix, short delays 1-5ms, high feedback
- Dark ambient wash: damping 0.7+, long delays, feedback 0.9+, high diffusion
- Chorus reverb: mod_master_rate 1-3 Hz, mod_depth_delay 3-8 samples, mod_correlation 0.3-0.6
"""

LOSSY_GUIDE = """You are an expert audio engineer tuning a codec artifact emulator (lossy audio effect).

SIGNAL CHAIN: Input -> Spectral Loss (STFT) -> Crush/Decimate -> Packets -> Filter -> Verb -> Gate -> Limiter -> Wet/Dry Mix -> Output

PARAMETERS AND RANGES:

Spectral Loss:
- mode: 0=Standard (quantize+gate spectral bins), 1=Inverse (residual — everything Standard discards), 2=Jitter (random phase noise, keeps magnitudes)
- loss (0.0-1.0): Destruction amount. 0=clean, 0.5=noticeable degradation, 1.0=heavily destroyed.
- speed (0.0-1.0): FFT window size. 0=slow/4096 (smooth), 1=fast/256 (glitchy). Controls rate of spectral variation.
- global_amount (0.0-1.0): Master intensity multiplier for all spectral processing.
- phase_loss (0.0-1.0): Phase quantization. 0=off, higher=more smeared/phasey.
- quantizer: 0=uniform (classic), 1=compand (MP3-style power-law codec).
- pre_echo (0.0-1.0): Boosts loss before transients, mimicking MP3 pre-echo artifacts.
- noise_shape (0.0-1.0): Envelope-following quantization — coarser in quiet bands.
- weighting (0.0-1.0): 0=equal frequency weighting, 1=psychoacoustic ATH model (like real codecs).

Crush (time-domain):
- crush (0.0-1.0): Bitcrusher. 0=16-bit (clean), 1=4-bit (destroyed).
- decimate (0.0-1.0): Sample rate reduction. 0=full rate, 1=extreme aliasing.

Packets:
- packets: 0=Clean, 1=Packet Loss (dropouts), 2=Packet Repeat (stutters).
- packet_rate (0.0-1.0): Probability of entering bad state (dropout/repeat).
- packet_size (5.0-200.0 ms): Chunk length for packet processing.

Filter:
- filter_type: 0=Bypass, 1=Bandpass, 2=Notch.
- filter_freq (20.0-20000.0 Hz): Center frequency.
- filter_width (0.0-1.0): 0=narrow/high-Q (resonant), 1=wide/low-Q (gentle).
- filter_slope: 0=6dB/oct (gentle), 1=24dB/oct (steep), 2=96dB/oct (brick wall).

Effects:
- verb (0.0-1.0): Lo-fi Schroeder reverb mix. Intentionally cheap and metallic.
- decay (0.0-1.0): Reverb decay length.
- freeze: 0=off, 1=on. Captures and holds spectral snapshot.
- freeze_mode: 0=Slushy (slowly evolving freeze), 1=Solid (static freeze).
- freezer (0.0-1.0): Frozen/live blend. 0=fully live, 1=fully frozen.
- gate (0.0-1.0): Noise gate threshold. 0=off.

Output:
- wet_dry (0.0-1.0): 0=dry, 1=wet.

RECIPES:
- Underwater/streaming: loss 0.7-0.9, speed 0.0, standard mode, no crush
- Glitchy digital: loss 0.5, speed 0.8-1.0, packet loss, crush 0.3
- Lo-fi radio: loss 0.4, bandpass filter at 800-2000Hz, verb 0.2, decimate 0.3
- Frozen texture: freeze on, slushy mode, loss 0.5, verb 0.3
- Extreme destruction: loss 1.0, crush 0.6, decimate 0.5, packet repeat
"""
```

### Success Criteria:

#### Automated Verification:
- [ ] File exists at `gui/llm_guide_text.py`
- [ ] Can import: `python -c "from gui.llm_guide_text import REVERB_GUIDE, LOSSY_GUIDE; print('OK')"`

---

## Phase 2: LLMTuner Core

### Overview
The shared `LLMTuner` class that handles Claude SDK interaction, threading, validation, and param clamping. Both GUIs will use this.

### Changes Required:

#### 1. Add dependency
**File**: `pyproject.toml`
Add `claude-agent-sdk` to dependencies.

#### 2. New file: `gui/llm_tuner.py`

**File**: `gui/llm_tuner.py` (new)

The class needs to:
- Run an asyncio event loop in a daemon thread (bridges async SDK with synchronous tkinter)
- Lazily create a `ClaudeSDKClient` on first prompt
- Build a system prompt from the guide text + current params
- Send the user prompt, collect the response
- Validate returned JSON: discard unknown keys, clamp numeric values to `PARAM_RANGES`, cast list elements for per-node params
- Merge result with current params (partial updates)
- Call back to the GUI on the main thread via `root.after()`
- Track pre-LLM params for undo
- Support "New Session" (disconnect old client, next prompt creates fresh one)

```python
class LLMTuner:
    def __init__(self, guide_text, param_ranges, default_params_fn, root):
        """
        guide_text: REVERB_GUIDE or LOSSY_GUIDE string
        param_ranges: PARAM_RANGES dict from engine/params.py
        default_params_fn: callable that returns default_params() dict (for key validation)
        root: tkinter root (for root.after scheduling)
        """

    def send_prompt(self, user_text, current_params, on_success, on_error):
        """Non-blocking. Runs async SDK call in background thread.
        on_success(merged_params: dict, response_text: str) — called on main thread
        on_error(error_msg: str) — called on main thread
        """

    def undo(self):
        """Returns the pre-LLM params dict, or None if nothing to undo."""

    def reset_session(self):
        """Disconnect current client. Next send_prompt creates a fresh session."""

    def shutdown(self):
        """Stop the event loop thread. Call on GUI close."""
```

Key implementation details:

**System prompt** (set once per session via `ClaudeAgentOptions.system_prompt`):
```
{guide_text}

RULES:
- Return ONLY a valid JSON object with parameter key-value pairs.
- You may include all parameters or just the ones you want to change.
- Missing keys will keep their current values.
- Stay within the documented ranges.
- For per-node parameters (arrays of 8), provide all 8 values.
- For integer choice params (mode, waveform, etc), use the integer value.
```

**Per-prompt user message** (includes current params for context):
```
Current parameters:
{json.dumps(current_params, indent=2)}

User request: {user_text}
```

**SDK options**:
```python
ClaudeAgentOptions(
    model="claude-opus-4-6",
    system_prompt=system_prompt,
    max_turns=1,
    allowed_tools=[],
    output_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                # All params listed here with their types
                # Scalars: {"type": "number"} or {"type": "integer"} or {"type": "string"}
                # Per-node arrays: {"type": "array", "items": {"type": "number"}}
            },
            "required": [],  # Everything optional — partial updates
            "additionalProperties": False,
        }
    }
)
```

The schema is auto-generated from `default_params()` and `PARAM_RANGES` at init time — inspect each value's type to produce the right JSON schema property, and embed the range + a brief description in the `"description"` field of each property. Example:

```json
"feedback_gain": {
    "type": "number",
    "description": "Range 0.0-2.0. Energy recirculation. 0=no reverb, 0.85=medium, >1.0 explodes without saturation."
}
"delay_times": {
    "type": "array",
    "items": {"type": "integer"},
    "description": "8 values, range 1-13230 samples. Delay lengths per node. Prime-ish ratios for density."
}
```

Claude sees these descriptions during generation, so it stays in range naturally. Clamping is just a safety net. No manual schema maintenance needed — adding a param to `default_params()` auto-updates the schema.

We use `output_format` structured output with a JSON schema. All properties listed, `"required": []` (empty — everything optional so Claude can return partial updates). `"additionalProperties": false` is required by the API. Note: `minItems`/`maxItems` and numeric `minimum`/`maximum` are NOT enforced by the schema — we still clamp values and validate array lengths in post-processing.

**Validation** (post structured output — JSON is guaranteed valid, but values may be out of range):
- Schema guarantees only known keys with correct types — no need to discard unknown keys
- For keys in `PARAM_RANGES`: clamp scalar values to (min, max); for list values, clamp each element
- For list params (delay_times, damping_coeffs, etc.): verify length == 8, truncate or pad with defaults if needed
- Merge: `current_params.copy()` then `.update(validated)`

**Undo**: Before applying new params, stash `current_params` in `self._undo_params`. `undo()` returns it.

**Error handling**: Catch any exception from the SDK call, format as string, call `on_error(str(e))`.

### Success Criteria:

#### Automated Verification:
- [ ] File exists at `gui/llm_tuner.py`
- [ ] Can import: `python -c "from gui.llm_tuner import LLMTuner; print('OK')"`
- [ ] `uv run --prerelease=allow python -c "import claude_agent_sdk; print('OK')"` succeeds after dependency install

---

## Phase 3: Reverb GUI Integration

### Overview
Add the AI prompt box to the bottom of the Parameters page in `gui/gui.py`.

### Changes Required:

#### 1. Add AI prompt widgets to Parameters page
**File**: `gui/gui.py`

In `_build_params_page()`, after the existing two-column layout (left + right frames that are packed into `inner`), add a new frame at the bottom of `inner` for the AI prompt. Since `inner` is the scrollable content frame, the AI box scrolls with everything else.

Widget structure:
```
inner (existing scrollable frame)
├── left  (existing - global params, matrix, XY, modulation)
├── right (existing - per-node sliders)
└── ai_frame (NEW - spans full width below both columns)
    ├── separator
    ├── label "AI Tuner"
    ├── input_frame
    │   ├── text entry (2-3 lines tall, multi-line)
    │   └── button_frame
    │       ├── "Ask Claude" button
    │       ├── "Undo" button
    │       └── "New Session" button
    ├── response_text (read-only Text widget, ~4 lines, shows Claude's explanation)
    └── ai_status label ("Ready" / "Thinking..." / "Applied 12 params" / error)
```

The text entry should submit on Shift+Enter (or a button click). Plain Enter adds a newline (multi-line input is useful for detailed prompts).

#### 2. Add methods to `ReverbGUI`
**File**: `gui/gui.py`

```python
def _build_ai_prompt(self, parent):
    """Build the AI tuner widgets into the given parent frame."""

def _on_ask_claude(self):
    """Collect current params, send prompt to LLMTuner, disable input while waiting."""
    current = self._read_params_from_ui()
    prompt = self.ai_input.get("1.0", "end-1c").strip()
    if not prompt:
        return
    self.ai_status_var.set("Thinking...")
    self.ai_ask_btn.configure(state="disabled")
    self.llm.send_prompt(prompt, current, self._on_claude_success, self._on_claude_error)

def _on_claude_success(self, merged_params, response_text):
    """Called on main thread. Apply params, show response, auto-play."""
    self._write_params_to_ui(merged_params)
    self.ai_response.configure(state="normal")
    self.ai_response.delete("1.0", "end")
    self.ai_response.insert("1.0", response_text)
    self.ai_response.configure(state="disabled")
    self.ai_status_var.set(f"Applied params")
    self.ai_ask_btn.configure(state="normal")
    self.ai_input.delete("1.0", "end")
    self._on_play()  # auto-play

def _on_claude_error(self, error_msg):
    """Called on main thread. Show error in status."""
    self.ai_status_var.set(f"Error: {error_msg}")
    self.ai_ask_btn.configure(state="normal")

def _on_ai_undo(self):
    """Revert to pre-LLM params."""
    prev = self.llm.undo()
    if prev:
        self._write_params_to_ui(prev)
        self.ai_status_var.set("Reverted to previous params")
        self._on_play()

def _on_ai_new_session(self):
    """Reset Claude conversation context."""
    self.llm.reset_session()
    self.ai_response.configure(state="normal")
    self.ai_response.delete("1.0", "end")
    self.ai_response.configure(state="disabled")
    self.ai_status_var.set("New session started")
```

#### 3. Initialize LLMTuner in `__init__`
**File**: `gui/gui.py`

```python
from gui.llm_guide_text import REVERB_GUIDE
from gui.llm_tuner import LLMTuner

# In __init__, after self._build_ui():
self.llm = LLMTuner(
    guide_text=REVERB_GUIDE,
    param_ranges=PARAM_RANGES,
    default_params_fn=default_params,
    root=self.root,
)
```

Add import of `PARAM_RANGES` from `engine.params` (already imported: `default_params, SR`).

### Success Criteria:

#### Automated Verification:
- [ ] `uv run --prerelease=allow python -c "from gui.gui import ReverbGUI; print('OK')"` imports without error

#### Manual Verification:
- [ ] Launch reverb GUI: `uv run --prerelease=allow python gui/gui.py`
- [ ] AI prompt box visible at bottom of Parameters page
- [ ] Type "dark cathedral reverb", click Ask Claude, sliders update and audio plays
- [ ] Type "brighter" as follow-up, Claude adjusts incrementally
- [ ] Undo reverts to previous params
- [ ] New Session clears context

---

## Phase 4: Lossy GUI Integration

### Overview
Same AI prompt box, added to the bottom of the Parameters page in `lossy/gui/gui.py`.

### Changes Required:

#### 1. Add AI prompt widgets to Parameters page
**File**: `lossy/gui/gui.py`

In `_build_params_page()`, after all the existing parameter sections, add the AI prompt frame at the bottom of the scrollable content. The lossy GUI uses `self.params_frame` with a scrollable canvas — add the AI box at the bottom of the inner frame.

Same widget structure as the reverb GUI (text entry + buttons + response + status).

#### 2. Add methods to `LossyGUI`
**File**: `lossy/gui/gui.py`

Same methods as reverb: `_build_ai_prompt`, `_on_ask_claude`, `_on_claude_success`, `_on_claude_error`, `_on_ai_undo`, `_on_ai_new_session`.

The only difference is `_on_claude_success` calls `self._on_play()` which is the lossy version.

#### 3. Initialize LLMTuner in `__init__`
**File**: `lossy/gui/gui.py`

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.llm_guide_text import LOSSY_GUIDE
from gui.llm_tuner import LLMTuner

# In __init__:
self.llm = LLMTuner(
    guide_text=LOSSY_GUIDE,
    param_ranges=PARAM_RANGES,
    default_params_fn=default_params,
    root=self.root,
)
```

The lossy GUI already has `sys.path.insert(0, ...)` pointing to the lossy directory. We need the parent (`reverb/`) on the path too so `from gui.llm_guide_text` resolves. The lossy `main.py` already adds its own directory; we add the reverb root for the shared import.

### Success Criteria:

#### Automated Verification:
- [ ] `uv run --prerelease=allow python -c "from lossy.gui.gui import LossyGUI; print('OK')"` imports without error

#### Manual Verification:
- [ ] Launch lossy GUI: `cd lossy && uv run --prerelease=allow python main.py`
- [ ] AI prompt box visible at bottom of Parameters page
- [ ] Type "underwater streaming audio", click Ask Claude, sliders update and audio plays
- [ ] Follow-up: "more glitchy", Claude adjusts
- [ ] Undo and New Session work

---

## Phase 5: Smoke Test & Polish

### Overview
End-to-end verification, minor tweaks.

### Steps:
1. Install dependency: `uv add claude-agent-sdk`
2. Launch reverb GUI, test full flow
3. Launch lossy GUI, test full flow
4. Verify multi-turn conversation works (Claude remembers context)
5. Verify error display when Claude is unavailable
6. Fix any issues found during testing

### Success Criteria:

#### Manual Verification:
- [ ] Reverb: prompt -> params applied -> audio plays -> follow-up -> undo -> new session
- [ ] Lossy: prompt -> params applied -> audio plays -> follow-up -> undo -> new session
- [ ] Error case: if Claude is unavailable, status shows error message (not a crash)
- [ ] GUI remains responsive while Claude is thinking (no freeze)

---

## Testing Strategy

### Manual Testing Steps:
1. "Give me a dark cathedral reverb" → verify feedback high, damping high, long delays
2. "Make it brighter" → verify damping decreases, other params stable
3. "Add slow chorus modulation" → verify mod_master_rate and mod_depth_delay set
4. Click Undo → verify reverts to step 2 params
5. Click New Session → type "completely different: metallic resonator" → verify fresh generation
6. Same flow for lossy: "underwater lo-fi", "more destruction", "add packet loss", undo, new session

## Performance Considerations

- Claude SDK call takes 5-30 seconds. GUI stays responsive because the call runs in a background thread.
- Only one request at a time — disable the "Ask Claude" button while waiting.
- The asyncio event loop thread is a daemon thread, so it dies when the GUI closes.

## References

- Research: `thoughts/shared/research/2026-02-06-llm-parameter-tuner-design.md`
- Reverb params: `engine/params.py`
- Lossy params: `lossy/engine/params.py`
- Reverb GUI: `gui/gui.py`
- Lossy GUI: `lossy/gui/gui.py`
