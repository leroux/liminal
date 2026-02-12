"""Synthesis engine base class and registry."""

from abc import ABC, abstractmethod
import numpy as np


class Engine(ABC):
    """Base class for all synthesis engines."""

    name: str = "Base"
    description: str = ""

    @abstractmethod
    def render(self, magnitude: np.ndarray, phase: np.ndarray | None,
               sr: int, n_fft: int, hop_length: int) -> np.ndarray:
        """Render magnitude spectrogram to audio.

        Args:
            magnitude: (n_freq_bins, n_frames) float32, 0-1
            phase: (n_freq_bins, n_frames) float32 or None
            sr: sample rate
            n_fft: FFT size
            hop_length: hop size

        Returns:
            audio: 1D float32 array
        """
        ...

    def get_params(self) -> dict:
        """Return engine-specific configurable parameters."""
        return {}


# Engine registry - populated by imports
_ENGINE_REGISTRY: dict[str, type[Engine]] = {}


def register_engine(cls: type[Engine]) -> type[Engine]:
    _ENGINE_REGISTRY[cls.name] = cls
    return cls


def get_engine_names() -> list[str]:
    return list(_ENGINE_REGISTRY.keys())


def get_engine_class(name: str) -> type[Engine]:
    return _ENGINE_REGISTRY[name]


def create_engine(name: str) -> Engine:
    return _ENGINE_REGISTRY[name]()


def load_all_engines():
    """Import all engine modules to trigger registration."""
    from . import random_phase
    from . import additive
    from . import subtractive
    from . import karplus_strong
    from . import griffin_lim
    from . import granular
    from . import fm
    from . import wavetable
    from . import formant
    from . import phase_vocoder
    from . import spectral_filter
