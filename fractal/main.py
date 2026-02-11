#!/usr/bin/env python3
"""Fractal — audio fractalization effect."""

import logging
import tkinter as tk
from fractal.gui.gui import FractalGUI

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")


def main():
    root = tk.Tk()
    app = FractalGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
