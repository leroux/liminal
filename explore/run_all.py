"""Top-level script: generate synthetic inputs, then run effects on each input."""
import argparse
import glob
import os
import sys

# Add explore dir to path so generate/registry imports work
sys.path.insert(0, os.path.dirname(__file__))

from generate_inputs import generate_all as generate_inputs
from generate import generate_all as generate_effects, DEFAULT_TIMEOUT


EXPLORE_DIR = os.path.dirname(__file__)
INPUTS_DIR = os.path.join(EXPLORE_DIR, 'inputs')
OUTPUT_DIR = os.path.join(EXPLORE_DIR, 'output')


def main():
    parser = argparse.ArgumentParser(description='Run all effects on all test inputs')
    parser.add_argument('--max-variants', type=int, default=1200)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--include', type=str, default=None,
                       help='Comma-separated glob patterns to include (e.g. "a*,j*")')
    parser.add_argument('--exclude', type=str, default=None,
                       help='Comma-separated glob patterns to exclude')
    parser.add_argument('--inputs-only', action='store_true',
                       help='Only generate input WAVs, skip effects')
    parser.add_argument('--input', type=str, default=None,
                       help='Run on a single specific input WAV instead of all')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                       help=f'Per-variant timeout in seconds (default: {DEFAULT_TIMEOUT})')
    args = parser.parse_args()

    # Step 1: Generate synthetic inputs
    print("=== Generating synthetic inputs ===")
    generate_inputs(INPUTS_DIR)
    print()

    if args.inputs_only:
        return

    # Step 2: Collect all input WAVs
    input_files = sorted(glob.glob(os.path.join(INPUTS_DIR, '*.wav')))
    if args.input:
        # Single input mode
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} not found")
            sys.exit(1)
        input_files = [args.input]

    print(f"=== Processing {len(input_files)} inputs ===")
    for wav_path in input_files:
        name = os.path.splitext(os.path.basename(wav_path))[0]
        out_dir = os.path.join(OUTPUT_DIR, name)
        print(f"\n{'='*60}")
        print(f"  Input: {name}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")
        generate_effects(
            wav_path, out_dir,
            max_variants=args.max_variants,
            workers=args.workers,
            include=args.include,
            exclude=args.exclude,
            timeout=args.timeout,
        )

    print(f"\n=== Done. Outputs in {OUTPUT_DIR}/ ===")


if __name__ == '__main__':
    main()
