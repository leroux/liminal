#!/usr/bin/env python3
"""Fractal — audio fractalization effect."""

import logging
import tkinter as tk
from fractal.gui.gui import FractalGUI
from shared.macos import set_app_name

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")


def main():
    set_app_name("Fractal")
    root = tk.Tk()
    app = FractalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
