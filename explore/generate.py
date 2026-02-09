"""Main entry point: takes input WAV, generates all effect variants.

Uses multiprocessing to render across all CPU cores.
Per-variant timeout prevents slow effects from hanging the run.
"""
import argparse
import json
import os
import signal
import sys
import time
import fnmatch
import multiprocessing
from multiprocessing import cpu_count

import numpy as np
import soundfile as sf

from registry import discover_effects
from primitives import post_process, post_process_stereo
from manifest import write_manifest

DEFAULT_TIMEOUT = 120  # seconds per variant


def load_input(path):
    """Load input WAV as mono float32."""
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def make_param_description(params, max_len=50):
    """Create a brief param string for filenames."""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k[:6]}{v:.2g}".replace('.', ''))
        elif isinstance(v, int):
            parts.append(f"{k[:6]}{v}")
        elif isinstance(v, str):
            parts.append(f"{k[:4]}_{v[:6]}")
    desc = '_'.join(parts)
    return desc[:max_len].rstrip('_')


def safe_filename(s, max_len=90):
    """Sanitize string for use as filename."""
    s = s.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe = ''.join(c for c in s if c.isalnum() or c in '_-.')
    return safe[:max_len]


def category_from_id(effect_id):
    """Extract category from effect ID prefix."""
    cats = {
        'a': 'a_delay', 'b': 'b_reverb', 'c': 'c_modulation', 'd': 'd_distortion',
        'e': 'e_filter', 'f': 'f_dynamics', 'g': 'g_pitch_time', 'h': 'h_spectral',
        'i': 'i_granular', 'j': 'j_chaos_math', 'k': 'k_neural', 'l': 'l_convolution',
        'm': 'm_physical', 'n': 'n_lofi', 'o': 'o_spatial', 'p': 'p_envelope',
        'q': 'q_combo', 'r': 'r_misc',
    }
    prefix = effect_id[0].lower()
    return cats.get(prefix, 'other')


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _process_one(args):
    """Worker function for multiprocessing. Runs a single effect variant.

    Takes a tuple to be pickle-friendly:
        (effect_id, params, samples, sr, output_dir, timeout)
    Returns entry dict on success, None on failure.
    """
    effect_id, params, samples, sr, output_dir, timeout = args

    # Set per-variant timeout via SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    # Re-import in worker process (each process needs its own modules)
    from registry import discover_effects as _discover
    from primitives import post_process as _pp, post_process_stereo as _pps

    registry = _discover()
    if effect_id not in registry:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None

    effect_fn, _ = registry[effect_id]
    category = category_from_id(effect_id)
    effect_name = effect_id.replace('_', ' ').title()

    try:
        if params:
            result = effect_fn(samples.copy(), sr, **params)
        else:
            result = effect_fn(samples.copy(), sr)

        if result is None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return None
        if not np.all(np.isfinite(result)):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return None
        if np.max(np.abs(result)) > 1e6:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return None

        if result.ndim == 2:
            result = _pps(result, sr)
        else:
            result = _pp(result, sr)
    except _Timeout:
        signal.signal(signal.SIGALRM, old_handler)
        return 'TIMEOUT'
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None

    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    param_desc = make_param_description(params) if params else 'default'
    code = effect_id.split('_')[0]
    fname = safe_filename(f"{code}_{param_desc}.wav")

    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)
    fpath = os.path.join(cat_dir, fname)
    tmp_path = fpath + '.part'
    sf.write(tmp_path, result, sr, format='WAV')
    os.replace(tmp_path, fpath)

    rel_path = os.path.join(category, fname)
    return {
        'filename': rel_path,
        'effect_id': code,
        'effect_name': effect_name,
        'params_json': json.dumps(params),
        'category': category,
        'duration_sec': f"{len(result)/sr:.2f}",
        'sample_rate': sr,
    }


def generate_all(input_path, output_dir, max_variants=1200, include=None, exclude=None,
                 workers=None, timeout=DEFAULT_TIMEOUT):
    """Generate all effect variants from input audio using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)

    samples, sr = load_input(input_path)
    print(f"Input: {input_path}, {len(samples)} samples, {sr} Hz, {len(samples)/sr:.2f}s")

    registry = discover_effects()
    print(f"Discovered {len(registry)} effects")

    # Filter effects by include/exclude patterns
    effect_ids = sorted(registry.keys())
    if include:
        patterns = [p.strip() for p in include.split(',')]
        effect_ids = [eid for eid in effect_ids
                     if any(fnmatch.fnmatch(eid, p) for p in patterns)]
    if exclude:
        patterns = [p.strip() for p in exclude.split(',')]
        effect_ids = [eid for eid in effect_ids
                     if not any(fnmatch.fnmatch(eid, p) for p in patterns)]

    # Build work items: (effect_id, params, samples, sr, output_dir, timeout)
    # Skip items whose output file already exists
    work_items = []
    skipped = 0
    for effect_id in effect_ids:
        _, variants_fn = registry[effect_id]
        if variants_fn:
            try:
                variant_params = variants_fn()
            except Exception as e:
                print(f"  Warning: variants for {effect_id} failed: {e}")
                variant_params = [{}]
        else:
            variant_params = [{}]

        for params in variant_params:
            if len(work_items) + skipped >= max_variants:
                break
            # Pre-check if output already exists
            category = category_from_id(effect_id)
            param_desc = make_param_description(params) if params else 'default'
            code = effect_id.split('_')[0]
            fname = safe_filename(f"{code}_{param_desc}.wav")
            fpath = os.path.join(output_dir, category, fname)
            if os.path.exists(fpath):
                skipped += 1
                continue
            work_items.append((effect_id, params, samples, sr, output_dir, timeout))
        if len(work_items) + skipped >= max_variants:
            break

    if skipped > 0:
        print(f"Skipped {skipped} already-generated variants")

    n_workers = workers or max(1, cpu_count() - 1)
    print(f"Processing {len(work_items)} variants across {n_workers} workers "
          f"({timeout}s timeout, 1 CPU reserved)")

    total_start = time.time()
    entries = []
    done = 0
    timed_out = 0

    # chunksize=1: one slow variant won't block a batch
    # maxtasksperchild=4: recycle workers to reclaim memory from leaked state
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=n_workers, maxtasksperchild=4) as pool:
        for result in pool.imap_unordered(_process_one, work_items, chunksize=1):
            done += 1
            if result == 'TIMEOUT':
                timed_out += 1
            elif result is not None:
                entries.append(result)
            if done % 20 == 0 or done == len(work_items):
                elapsed = time.time() - total_start
                rate = done / elapsed
                eta = (len(work_items) - done) / rate if rate > 0 else 0
                timeout_str = f", {timed_out} timed out" if timed_out else ""
                print(f"  [{done}/{len(work_items)}] {len(entries)} ok{timeout_str}, "
                      f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - total_start
    timeout_str = f", {timed_out} timed out" if timed_out else ""
    print(f"\nGenerated {len(entries)} outputs in {elapsed:.1f}s "
          f"({len(entries)/elapsed:.1f} variants/sec{timeout_str})")

    # Sort by category then effect code for stable grouped order
    entries.sort(key=lambda e: (e['category'], e['effect_id'], e['filename']))
    write_manifest(output_dir, entries)

    return entries


def main():
    parser = argparse.ArgumentParser(description='Generate audio effect variants')
    parser.add_argument('input_wav', help='Input WAV file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--max-variants', type=int, default=1200)
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Number of worker processes (default: {max(1, cpu_count() - 1)} = all CPUs minus 1)')
    parser.add_argument('--include', type=str, default=None,
                       help='Comma-separated glob patterns to include (e.g. "a*,j*")')
    parser.add_argument('--exclude', type=str, default=None,
                       help='Comma-separated glob patterns to exclude (e.g. "q*")')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                       help=f'Per-variant timeout in seconds (default: {DEFAULT_TIMEOUT})')
    args = parser.parse_args()

    generate_all(args.input_wav, args.output_dir,
                max_variants=args.max_variants,
                workers=args.workers,
                include=args.include, exclude=args.exclude,
                timeout=args.timeout)


if __name__ == '__main__':
    main()
