#!/usr/bin/env python3
"""Lossy — codec artifact emulator."""

import logging
import tkinter as tk
from lossy.gui.gui import LossyGUI
from shared.macos import set_app_name

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")


def main():
    set_app_name("Lossy")
    root = tk.Tk()
    app = LossyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
