#!/usr/bin/env python3
"""Lossy — codec artifact emulator.

Run from the lossy/ directory:
    python main.py
"""

import sys
import os

# Add this directory to path so 'engine', 'gui', 'audio' are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from gui.gui import LossyGUI


def main():
    root = tk.Tk()
    app = LossyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
