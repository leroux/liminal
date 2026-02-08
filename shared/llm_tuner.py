"""LLM-assisted parameter tuner using the Claude Agent SDK.

Shared between both pedals (reverb + lossy). Each GUI instantiates LLMTuner
with its own guide text, param ranges, and default_params function.
"""

import asyncio
import base64
import json
import logging
import os
import re
import threading
import traceback

log = logging.getLogger(__name__)


def _bytes_to_base64(data):
    """Encode raw bytes to base64 string for API image blocks."""
    return base64.b64encode(data).decode("ascii")


class LLMTuner:
    """Bridges the async Claude Agent SDK with the synchronous tkinter GUI.

    Runs an asyncio event loop in a daemon thread. Calls to send_prompt()
    are non-blocking — results stream back via callbacks on the tkinter main thread.
    """

    def __init__(self, guide_text, param_descriptions, param_ranges, default_params_fn, root):
        """
        guide_text:          REVERB_GUIDE or LOSSY_GUIDE system prompt string
        param_descriptions:  dict of param_name -> description string for JSON schema
        param_ranges:        PARAM_RANGES dict from engine/params.py
        default_params_fn:   callable returning default_params() dict
        root:                tkinter root (for root.after scheduling)
        """
        self._guide_text = guide_text
        self._param_descriptions = param_descriptions
        self._param_ranges = param_ranges
        self._default_params_fn = default_params_fn
        self._root = root
        self._undo_params = None
        self._client = None
        self._busy = False
        self._prev_metrics = None   # last output metrics (for A/B delta)
        self._source_sent = False   # whether source info has been sent this session

        # Async event loop in a background daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _build_system_prompt(self):
        return self._guide_text + """

RULES:
- You can have a conversation with the user to understand what they want before
  committing parameter changes. Ask clarifying questions if needed.
- When you're ready to apply changes, include a ```json code block with a JSON
  object of parameter key-value pairs. This will be parsed and applied to the UI.
- If the user is just chatting or asking questions, respond normally without
  a JSON code block. Not every message needs parameter changes.
- You may include all parameters or just the ones you want to change.
- Missing keys will keep their current values.
- Stay within the documented ranges.
- For per-node parameters (arrays of 8), always provide all 8 values.
- For integer choice params (mode, waveform, etc), use the integer value.
- Include a brief text explanation of what you're changing and why.
"""

    def _build_options(self):
        from claude_agent_sdk import ClaudeAgentOptions
        return ClaudeAgentOptions(
            model="claude-opus-4-6",
            system_prompt=self._build_system_prompt(),
            max_turns=None,
            allowed_tools=["Read"],
        )

    def send_prompt(self, user_text, current_params, on_text, on_params, on_done, on_error,
                    metrics=None, source_metrics=None,
                    spectrogram_png=None, source_spectrogram_png=None):
        """Non-blocking. Runs async SDK call in background thread.

        on_text(text)              — streamed assistant text, called on tkinter main thread
        on_params(merged)          — called when structured output arrives (params applied)
        on_done()                  — called when response is complete
        on_error(error_msg)        — called on error
        metrics                    — optional output metrics dict from analyze()
        source_metrics             — optional input/source audio metrics dict
        spectrogram_png            — optional output spectrogram PNG bytes
        source_spectrogram_png     — optional source spectrogram PNG bytes
        """
        if self._busy:
            self._root.after(0, on_error, "Already processing a request")
            return
        self._busy = True
        self._undo_params = current_params.copy()
        asyncio.run_coroutine_threadsafe(
            self._async_send(user_text, current_params, on_text, on_params, on_done, on_error,
                             metrics=metrics, source_metrics=source_metrics,
                             spectrogram_png=spectrogram_png,
                             source_spectrogram_png=source_spectrogram_png),
            self._loop,
        )

    @staticmethod
    def _extract_json_block(text):
        """Extract a JSON object from a ```json code block, if present."""
        m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _strip_json_block(text):
        """Remove ```json code blocks from text for display."""
        return re.sub(r"```json\s*\n.*?```\n?", "", text, flags=re.DOTALL).strip()

    async def _async_send(self, user_text, current_params, on_text, on_params, on_done, on_error,
                          metrics=None, source_metrics=None,
                          spectrogram_png=None, source_spectrogram_png=None):
        from claude_agent_sdk import ResultMessage, AssistantMessage

        try:
            if self._client is None:
                from claude_agent_sdk import ClaudeSDKClient
                self._client = ClaudeSDKClient(options=self._build_options())
                await self._client.connect()

            # Build text prompt
            full_prompt = (
                f"Current parameters:\n{json.dumps(current_params, indent=2)}"
                f"\n\nUser request: {user_text}"
            )

            # Formatted audio metrics with A/B delta and dedup
            if metrics is not None:
                from shared.audio_features import format_features
                src = source_metrics if not self._source_sent else None
                features_text = format_features(metrics, self._prev_metrics, src)
                if features_text:
                    full_prompt += f"\n\n{features_text}"
                self._prev_metrics = metrics

            # Write spectrograms to tmp files so Claude can Read them
            spec_dir = os.path.join(os.getcwd(), ".tmp_spectrograms")
            os.makedirs(spec_dir, exist_ok=True)

            if not self._source_sent and source_spectrogram_png:
                src_path = os.path.join(spec_dir, "input.png")
                with open(src_path, "wb") as f:
                    f.write(source_spectrogram_png)
                full_prompt += (
                    f"\n\nInput audio spectrogram saved at: {src_path}"
                    f"\nUse the Read tool to view it."
                )

            if spectrogram_png:
                out_path = os.path.join(spec_dir, "output.png")
                with open(out_path, "wb") as f:
                    f.write(spectrogram_png)
                full_prompt += (
                    f"\n\nOutput audio spectrogram saved at: {out_path}"
                    f"\nUse the Read tool to view it."
                )

            self._source_sent = True

            log.debug("LLM prompt:\n%s", full_prompt)
            await self._client.query(full_prompt)

            async for msg in self._client.receive_response():
                log.debug("LLM msg: %s %r", type(msg).__name__, vars(msg) if hasattr(msg, '__dict__') else msg)

                if isinstance(msg, AssistantMessage):
                    for block in (msg.content or []):
                        if hasattr(block, 'text') and block.text:
                            raw_params = self._extract_json_block(block.text)
                            display_text = self._strip_json_block(block.text) if raw_params else block.text
                            if display_text:
                                self._root.after(0, on_text, display_text)
                            if raw_params:
                                validated = self._validate_and_clamp(raw_params)
                                merged = current_params.copy()
                                merged.update(validated)
                                self._root.after(0, on_params, merged)

            self._busy = False
            self._root.after(0, on_done)

        except Exception as e:
            traceback.print_exc()
            self._busy = False
            self._root.after(0, on_error, str(e))

    def _validate_and_clamp(self, raw):
        """Validate keys, clamp values to PARAM_RANGES, fix array lengths."""
        defaults = self._default_params_fn()
        valid_keys = set(defaults.keys())
        result = {}

        for key, value in raw.items():
            if key not in valid_keys:
                continue

            default_val = defaults[key]

            # Handle list params (per-node arrays)
            if isinstance(default_val, list):
                if not isinstance(value, list):
                    continue
                expected_len = len(default_val)
                # Pad or truncate to expected length
                if len(value) < expected_len:
                    value = value + default_val[len(value):]
                elif len(value) > expected_len:
                    value = value[:expected_len]
                # Clamp each element if range exists
                if key in self._param_ranges:
                    lo, hi = self._param_ranges[key]
                    value = [max(lo, min(hi, v)) for v in value]
                # Cast to match default type
                if default_val and isinstance(default_val[0], int):
                    value = [int(round(v)) for v in value]
                result[key] = value

            elif isinstance(default_val, int) and not isinstance(default_val, bool):
                try:
                    v = int(round(value))
                except (TypeError, ValueError):
                    continue
                if key in self._param_ranges:
                    lo, hi = self._param_ranges[key]
                    v = max(lo, min(hi, v))
                result[key] = v

            elif isinstance(default_val, float):
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if key in self._param_ranges:
                    lo, hi = self._param_ranges[key]
                    v = max(lo, min(hi, v))
                result[key] = v

            elif isinstance(default_val, str):
                result[key] = str(value)

            else:
                result[key] = value

        return result

    def undo(self):
        """Returns the pre-LLM params dict, or None if nothing to undo."""
        return self._undo_params

    def reset_session(self):
        """Disconnect current client. Next send_prompt creates a fresh session."""
        self._prev_metrics = None
        self._source_sent = False
        if self._client is not None:
            asyncio.run_coroutine_threadsafe(
                self._async_disconnect(), self._loop
            )

    async def _async_disconnect(self):
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def shutdown(self):
        """Stop the event loop thread. Call on GUI close."""
        self.reset_session()
        self._loop.call_soon_threadsafe(self._loop.stop)
