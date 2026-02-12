"""Source signal generators for subtractive synthesis and other engines."""

from .utils import generate_sawtooth, generate_white_noise, generate_pink_noise, generate_pulse_train

__all__ = ['generate_sawtooth', 'generate_white_noise', 'generate_pink_noise', 'generate_pulse_train']
