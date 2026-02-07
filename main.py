#!/usr/bin/env python3
"""Launch either pedal from the project root.

Usage:
    uv run python -m reverb.main    # reverb FDN
    uv run python -m lossy.main     # lossy codec emulator
"""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "lossy":
        from lossy.main import main
    else:
        from reverb.main import main
    main()
