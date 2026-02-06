---
date: "2026-02-06T16:00:00-05:00"
researcher: Claude
git_commit: 6c35439dfd38be45e7418825b4c595b7c8e21cf7
branch: main
repository: leroux/liminal
topic: "LLM-Assisted Parameter Tuner Design for Reverb and Lossy Pedals"
tags: [research, llm, gui, parameters, claude-agent-sdk, design]
status: complete
last_updated: "2026-02-06"
last_updated_by: Claude
last_updated_note: "Updated to use claude-agent-sdk approach per user direction"
---

# Research: LLM-Assisted Parameter Tuner Design

**Date**: 2026-02-06
**Git Commit**: 6c35439
**Branch**: main
**Repository**: leroux/liminal

## Research Question

Design an LLM-assisted parameter tuner for both pedals (Reverb FDN + Lossy codec emulator). The user wants a free-form text prompt within the GUI that calls Claude via the Claude Agent SDK, gets back a parameter JSON, and loads it into the GUI. Claude Code is already authenticated locally — no API key setup needed.

## Summary

Both pedals share an identical parameter architecture: a `params` dict defined in `engine/params.py`, round-tripped through the GUI via `_read_params_from_ui()` / `_write_params_to_ui()`, and persisted as JSON presets. This means a single integration pattern works for both.

**Chosen approach: `claude-agent-sdk` Python package.** The SDK wraps the local Claude Code CLI, which is already authenticated via the user's subscription. No API key needed. It provides native async Python API with structured JSON output support, multi-turn conversation via `ClaudeSDKClient`, and clean integration into the tkinter GUI via a background thread + asyncio event loop.

---

## Detailed Findings

### 1. Both Pedals Share the Same Parameter Architecture

#### Reverb FDN (`engine/params.py`)

- **`default_params()`** returns a flat dict with ~35 keys (delay_times, damping_coeffs, feedback_gain, modulation params, etc.)
- **`PARAM_RANGES`** defines min/max bounds for ML exploration
- Preset JSON format: flat dict matching the params schema exactly
- Example: `gui/presets/lush_chorus_room.json` — 24 key-value pairs

#### Lossy Codec Emulator (`lossy/engine/params.py`)

- **`default_params()`** returns a flat dict with ~30 keys (loss, speed, mode, crush, packets, filter, etc.)
- **`PARAM_RANGES`** defines min/max bounds
- Preset JSON format: identical pattern — flat dict
- Example: `lossy/gui/presets/underwater.json`

#### Shared Pattern for Both

```
JSON preset → default_params().update(preset) → _write_params_to_ui() → GUI sliders
GUI sliders → _read_params_from_ui() → params dict → render_*() engine / json.dump()
```

The integration point is clear: **produce a valid params dict as JSON → call `_write_params_to_ui()`**. This is exactly what preset loading already does.

---

### 2. Claude Agent SDK

#### Installation

```bash
pip install claude-agent-sdk
```

Requires Python 3.10+. The SDK bundles the Claude Code CLI internally.

#### Authentication

The SDK delegates auth to the local Claude Code CLI. Since Claude Code is already logged in via the user's subscription, **no API key or additional setup is needed**.

#### Key APIs

**`query()` — one-off calls (new session each time):**
```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def ask_claude(prompt: str):
    options = ClaudeAgentOptions(
        system_prompt="You are an audio DSP expert...",
        max_turns=1,
        allowed_tools=[],  # no tools needed, just text generation
    )
    async for message in query(prompt=prompt, options=options):
        # message types: AssistantMessage, ResultMessage, etc.
        pass
```

**`ClaudeSDKClient` — multi-turn conversation (maintains context):**
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

async with ClaudeSDKClient(options=options) as client:
    await client.query("make a dark ambient reverb")
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)

    # Follow-up — Claude remembers context
    await client.query("now make it brighter but keep the long tail")
    async for msg in client.receive_response():
        ...
```

**Structured output (JSON schema enforcement):**
```python
options = ClaudeAgentOptions(
    output_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "feedback_gain": {"type": "number"},
                "wet_dry": {"type": "number"},
                ...
            }
        }
    }
)
```

#### Why `ClaudeSDKClient` Over `query()`

For the parameter tuner, `ClaudeSDKClient` is the right choice because:
1. **Multi-turn refinement**: User says "make it darker" → "now add more modulation" → "less feedback". Claude remembers context.
2. **System prompt set once**: Configure the parameter schema once at session start.
3. **Session lifecycle**: Create client when GUI opens, close when GUI closes.

---

### 3. Integration Design

#### Flow

```
User types prompt in GUI text box → "make a dark cathedral reverb"
        │
        ▼
Background thread runs async call:
  ClaudeSDKClient.query(user_prompt)
        │
        ▼
Receive response, extract JSON from text
        │
        ▼
Validate keys, clamp values to PARAM_RANGES
        │
        ▼
Merge with current params (partial updates supported)
        │
        ▼
root.after(0, _write_params_to_ui, merged_params)  ← back on main thread
        │
        ▼
Optionally auto-render and play
```

#### System Prompt Design

The system prompt should include:
1. **The full parameter schema** with types, ranges, and descriptions
2. **The current parameter values** so Claude can make incremental adjustments
3. **Descriptions of what each parameter does sonically** (from the Guide tab content)
4. **Instructions to return ONLY a JSON object** matching the params schema
5. **Allowance for partial JSON** — only the keys the user wants changed (merged with current)

Example:
```
You are an expert audio DSP engineer tuning an 8-node FDN reverb.

RULES:
- Return ONLY a valid JSON object with parameter key-value pairs.
- You may include all parameters or just the ones you want to change.
- Missing keys will keep their current values.
- Stay within the documented ranges.

PARAMETER SCHEMA:
{schema with descriptions and ranges}

CURRENT VALUES:
{current params as JSON}
```

The user prompt is then just their free-form text: "give me a burial-style reverb with long tails" or "reduce the feedback a tiny bit".

---

### 4. Threading Model: Async SDK in tkinter

tkinter runs a synchronous mainloop. The SDK is async. Solution: run an asyncio event loop in a dedicated background thread.

```python
import asyncio
import threading

class LLMTuner:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = None  # ClaudeSDKClient, created lazily

    def send_prompt(self, prompt, current_params, callback):
        """Non-blocking. Calls callback(result_dict) on the main thread when done."""
        asyncio.run_coroutine_threadsafe(
            self._async_send(prompt, current_params, callback),
            self._loop
        )

    async def _async_send(self, prompt, current_params, callback):
        if self._client is None:
            self._client = ClaudeSDKClient(options=self._build_options())
            await self._client.connect()

        # Include current params in the prompt
        full_prompt = f"Current params:\n{json.dumps(current_params, indent=2)}\n\nUser request: {prompt}"
        await self._client.query(full_prompt)

        response_text = ""
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        params = self._parse_and_validate(response_text)
        # Schedule callback on tkinter main thread
        root.after(0, callback, params, response_text)
```

---

### 5. GUI Integration Points

#### Reverb GUI (`gui/gui.py`)

- **`_build_ui()`** (line 55): Where tabs are created — add "AI" tab
- **`_write_params_to_ui(p)`** (line 1225): Already handles applying a full params dict to all sliders
- **`_read_params_from_ui()`** (line 1197): Already handles collecting current params
- **`_on_load_preset()`** (line 1712): Pattern to follow — loads JSON, merges with defaults, writes to UI
- **`_on_play()`**: Can be called after applying LLM params for immediate feedback

#### Lossy GUI (`lossy/gui/gui.py`)

Identical integration points:
- **`_write_params_to_ui(p)`** (line 672): Applies params dict to all widgets
- **`_read_params_from_ui()`** (line 659): Collects current params
- **`_on_load_preset()`** (line 319): Same merge-with-defaults pattern
- **`_on_play()`**: Render and play after applying

---

### 6. Existing Parameter Documentation (For System Prompt)

#### Reverb — Guide Tab Content (`gui/gui.py:432-690`)

The Guide tab already contains rich parameter descriptions:
- Feedback Gain: "Controls how much energy recirculates... 0=no reverb, 0.85=medium, >1.0=unstable"
- Wet/Dry Mix: "0=dry only, 0.5=equal blend, 1.0=100% reverb"
- Diffusion: "Smears input through allpass filters... 0=bypass, 0.5+=heavy smearing"
- Saturation: "tanh soft-clipping... prevents explosion when feedback > 1.0"
- Modulation params with timescale descriptions
- Recipes section with practical combinations

#### Lossy — Guide Tab Content (`lossy/gui/gui.py:498-652`)

Similar rich descriptions:
- Loss: "0=clean, 0.5=noticeable degradation, 1.0=heavily destroyed"
- Speed: "Rate of spectral variation... slow=ambient, fast=glitchy"
- Mode descriptions (Standard/Inverse/Jitter)
- Signal chain documentation

#### PARAM_RANGES (both pedals)

Machine-readable min/max bounds. Used for clamping LLM output.
- Reverb: `engine/params.py:110-133`
- Lossy: `lossy/engine/params.py:97-120`

---

### 7. JSON Parsing Strategy

Claude's response will be text that *contains* JSON (or pure JSON if structured output works). Need robust extraction:

1. **If using structured output** (`output_format` with JSON schema): response is guaranteed valid JSON
2. **Fallback text parsing**:
   - Try direct `json.loads()` on the full response
   - Regex extract JSON block from markdown code fences: `` ```json ... ``` ``
   - Find first `{` to last `}` as fallback
3. **Validate** extracted dict keys against known parameter names, discard unknown keys
4. **Clamp values** to PARAM_RANGES bounds
5. **Merge** with current params (not defaults) for incremental changes

---

### 8. Conversation History for Iterative Refinement

The `ClaudeSDKClient` maintains conversation state automatically. Each `client.query()` call is a follow-up in the same session. Claude remembers:
- The parameter schema from the system prompt
- Previous parameter sets it generated
- Previous user feedback ("too bright", "more of that", etc.)

This enables natural multi-turn workflows:
```
User: "give me a large cathedral"
Claude: {feedback_gain: 0.93, damping_coeffs: [0.25]*8, ...}
User: "darker, like it's underground"
Claude: {damping_coeffs: [0.65]*8, ...}  ← only changes what's needed
User: "perfect, now add some slow modulation"
Claude: {mod_master_rate: 0.15, mod_depth_delay: [8.0]*8, ...}
```

---

## Architecture Documentation

### Proposed Component: `LLMTuner` (shared between both pedals)

```
gui/llm_tuner.py  (new file, shared logic)
├── class LLMTuner
│   ├── __init__(param_schema, param_ranges, guide_text, root)
│   ├── _build_options() -> ClaudeAgentOptions
│   ├── send_prompt(user_prompt, current_params, callback) -> None  [non-blocking]
│   ├── _async_send(prompt, current_params, callback) -> None       [async coroutine]
│   ├── _parse_response(raw_text) -> dict
│   ├── _validate_and_clamp(params_dict) -> dict
│   ├── reset_session() -> None                                     [start fresh conversation]
│   └── shutdown() -> None                                          [cleanup]
```

Both GUIs instantiate `LLMTuner` with their respective schemas:

```python
# In reverb gui/gui.py
from gui.llm_tuner import LLMTuner
self.llm = LLMTuner(
    param_schema=default_params(),
    param_ranges=PARAM_RANGES,
    guide_text=REVERB_GUIDE_TEXT,
    root=self.root
)

# In lossy gui/gui.py
from gui.llm_tuner import LLMTuner  # same module, different schema
self.llm = LLMTuner(
    param_schema=default_params(),
    param_ranges=PARAM_RANGES,
    guide_text=LOSSY_GUIDE_TEXT,
    root=self.root
)
```

### GUI Widget: "AI" Tab

For both pedals, add a new notebook tab with:

```
┌─────────────────────────────────────────────┐
│  AI Parameter Tuner                         │
├─────────────────────────────────────────────┤
│                                             │
│  [Text input area - multi-line]             │
│  "make a dark ambient wash with slow mod"   │
│                                             │
├─────────────────────────────────────────────┤
│  [Ask Claude]  [Apply & Play]  [Undo]       │
│  [New Session]  [Save as Preset]            │
├─────────────────────────────────────────────┤
│                                             │
│  Claude's Response:                         │
│  ┌─────────────────────────────────────┐    │
│  │ Here's a dark ambient configuration │    │
│  │ with heavy damping and slow         │    │
│  │ modulation for evolving texture...  │    │
│  │                                     │    │
│  │ {                                   │    │
│  │   "feedback_gain": 0.92,            │    │
│  │   "damping_coeffs": [0.7, ...],     │    │
│  │   ...                               │    │
│  │ }                                   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  Status: Applied 12 parameters              │
└─────────────────────────────────────────────┘
```

**Interaction flow:**
1. User types prompt, presses Enter or clicks "Ask Claude"
2. Status shows "Thinking..." with a spinner
3. Claude responds with explanation + JSON
4. Response displayed in text area
5. Parameters auto-applied to sliders (or click "Apply & Play")
6. User can iterate: "more feedback", "less modulation", etc.
7. "Undo" reverts to pre-LLM params
8. "New Session" resets conversation context
9. "Save as Preset" saves the LLM-generated params

---

## Code References

- `engine/params.py:10-106` — Reverb default_params() and full schema
- `engine/params.py:110-133` — Reverb PARAM_RANGES
- `gui/gui.py:1197-1223` — Reverb _read_params_from_ui()
- `gui/gui.py:1225-1262` — Reverb _write_params_to_ui()
- `gui/gui.py:1712-1725` — Reverb preset loading pattern
- `gui/gui.py:432-690` — Reverb Guide tab text (parameter descriptions)
- `lossy/engine/params.py:38-94` — Lossy default_params() and full schema
- `lossy/engine/params.py:97-120` — Lossy PARAM_RANGES
- `lossy/gui/gui.py:659-670` — Lossy _read_params_from_ui()
- `lossy/gui/gui.py:672-682` — Lossy _write_params_to_ui()
- `lossy/gui/gui.py:319-332` — Lossy preset loading pattern
- `lossy/gui/gui.py:498-652` — Lossy Guide tab text (parameter descriptions)

## Decisions (Resolved)

1. **Auto-apply**: Yes — params auto-apply as soon as Claude responds. No confirmation step.
2. **Auto-play**: Yes — auto-render and play after applying.
3. **Model**: Hardcode Claude Opus 4.6 (`claude-opus-4-6`). No model picker.
4. **Structured output**: Use `output_format` JSON schema. No regex fallback parsing.
5. **Rate limiting**: Show warning in status: "Claude unavailable — rate limited". No retry logic.
6. **Errors**: Just show the error in status. No offline fallback or graceful degradation.
7. **Guide text**: Hardcode a condensed version in a **shared location** (`gui/llm_guide_text.py` or similar) so both pedals import from one spot. No duplication.
