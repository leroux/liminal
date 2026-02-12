#!/usr/bin/env python3
"""Reverb — 8-node FDN reverb."""

import logging
import tkinter as tk
from reverb.gui.gui import ReverbGUI
from shared.macos import set_app_name

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")


def main():
    set_app_name("FDN Reverb")
    root = tk.Tk()
    root.geometry("1050x850")
    ReverbGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
