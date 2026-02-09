"""Writes manifest.csv alongside outputs."""
import csv
import json
import os


def write_manifest(output_dir, entries):
    """Write manifest.csv with all generated output info.

    entries: list of dicts with keys:
        filename, effect_id, effect_name, params_json, category, duration_sec, sample_rate
    """
    path = os.path.join(output_dir, 'manifest.csv')
    fieldnames = ['filename', 'effect_id', 'effect_name', 'params_json',
                  'category', 'duration_sec', 'sample_rate']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
    print(f"Manifest written: {path} ({len(entries)} entries)")
