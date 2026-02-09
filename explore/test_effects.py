"""Quick smoke test: runs every effect on a synthetic signal."""
import numpy as np
import time
import sys
from registry import discover_effects
from primitives import post_process, post_process_stereo


def make_test_signal(sr=44100, duration=2.0):
    """Generate test signal: sine sweep + click + noise burst + silence + tone.

    Exercises transient response, frequency response, and silence handling.
    """
    n = int(sr * duration)
    parts = []

    # Sine sweep 100Hz to 8000Hz (first 0.8s)
    sweep_n = int(sr * 0.8)
    t = np.linspace(0, 0.8, sweep_n, dtype=np.float32)
    phase = 2 * np.pi * 100 * 0.8 * (np.exp(t / 0.8 * np.log(8000 / 100)) - 1) / np.log(8000 / 100)
    parts.append(0.5 * np.sin(phase).astype(np.float32))

    # Click (impulse)
    click = np.zeros(int(sr * 0.1), dtype=np.float32)
    click[0] = 0.9
    parts.append(click)

    # Noise burst
    rng = np.random.default_rng(42)
    parts.append(0.3 * rng.standard_normal(int(sr * 0.3)).astype(np.float32))

    # Silence
    parts.append(np.zeros(int(sr * 0.2), dtype=np.float32))

    # Short tonal segment (A440)
    tone_n = int(sr * 0.4)
    t2 = np.linspace(0, 0.4, tone_n, dtype=np.float32)
    parts.append(0.5 * np.sin(2 * np.pi * 440 * t2).astype(np.float32))

    # Silence pad
    parts.append(np.zeros(int(sr * 0.2), dtype=np.float32))

    signal = np.concatenate(parts)
    return signal[:n]


def test_all_effects(verbose=True):
    """Run every discovered effect with default params. Returns (passed, failed, skipped)."""
    sr = 44100
    signal = make_test_signal(sr, 2.0)

    registry = discover_effects()
    passed = []
    failed = []
    skipped = []

    for effect_id in sorted(registry.keys()):
        effect_fn, variants_fn = registry[effect_id]
        try:
            t0 = time.time()
            result = effect_fn(signal.copy(), sr)
            elapsed = time.time() - t0

            # Check output
            assert result is not None, "Output is None"
            assert isinstance(result, np.ndarray), f"Output is {type(result)}, not ndarray"
            assert np.all(np.isfinite(result)), "Output contains NaN or Inf"
            assert len(result) > 0, "Output is empty"

            peak = np.max(np.abs(result))
            assert peak < 1e6, f"Peak amplitude too large: {peak}"

            # Post-process check
            if result.ndim == 2:
                pp = post_process_stereo(result, sr)
            else:
                pp = post_process(result, sr)
            assert np.all(np.isfinite(pp)), "Post-processed output contains NaN/Inf"

            if verbose:
                print(f"  PASS  {effect_id:40s}  {elapsed:6.3f}s  peak={peak:.4f}")
            passed.append(effect_id)

        except Exception as e:
            if verbose:
                print(f"  FAIL  {effect_id:40s}  {e}")
            failed.append((effect_id, str(e)))

    print(f"\nResults: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("Failed effects:")
        for eid, err in failed:
            print(f"  {eid}: {err}")

    return passed, failed, skipped


def test_variants(max_per_effect=2, verbose=True):
    """Test variant generation for all effects."""
    sr = 44100
    signal = make_test_signal(sr, 2.0)
    registry = discover_effects()

    total_variants = 0
    variant_failures = []

    for effect_id in sorted(registry.keys()):
        effect_fn, variants_fn = registry[effect_id]
        if variants_fn is None:
            continue

        try:
            variants = variants_fn()
            assert isinstance(variants, list), "variants_fn must return a list"
            assert len(variants) >= 2, f"Need at least 2 variants, got {len(variants)}"

            for i, params in enumerate(variants[:max_per_effect]):
                result = effect_fn(signal.copy(), sr, **params)
                assert result is not None and np.all(np.isfinite(result))
                total_variants += 1

        except Exception as e:
            variant_failures.append((effect_id, str(e)))
            if verbose:
                print(f"  VFAIL {effect_id}: {e}")

    print(f"\nVariant tests: {total_variants} tested, {len(variant_failures)} failures")
    return total_variants, variant_failures


if __name__ == '__main__':
    print("=" * 70)
    print("Audio Effects Smoke Test")
    print("=" * 70)

    print("\n--- Testing all effects with defaults ---")
    passed, failed, skipped = test_all_effects()

    if '--variants' in sys.argv:
        print("\n--- Testing variants ---")
        test_variants()

    sys.exit(1 if failed else 0)
