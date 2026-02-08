#!/usr/bin/env python3
"""Launch lossy GUI backed by Rust DSP engine."""

import logging
import os
import sys
import tkinter as tk

# Add project root to path for shared/ imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, project_root)

from lossy_gui import LossyGUI

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")


def main():
    root = tk.Tk()
    root.title("Lossy (Rust DSP)")
    app = LossyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
