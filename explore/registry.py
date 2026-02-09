"""Auto-discovers all effects and their variants, maps IDs to functions."""
import importlib
import inspect


EFFECT_MODULES = [
    'a_delay', 'b_reverb', 'c_modulation', 'd_distortion', 'e_filter',
    'f_dynamics', 'g_pitch_time', 'h_spectral', 'i_granular',
    'j_chaos_math', 'k_neural', 'l_convolution', 'm_physical',
    'n_lofi', 'o_spatial', 'p_envelope', 'q_combo', 'r_misc',
]


def discover_effects():
    """Returns dict: {effect_id: (effect_fn, variants_fn)}

    effect_id is the part after 'effect_', e.g. 'a001_simple_delay'.
    variants_fn may be None if no variants function exists.
    """
    registry = {}
    for mod_name in EFFECT_MODULES:
        try:
            mod = importlib.import_module(f'effects.{mod_name}')
        except ImportError as e:
            print(f"Warning: could not import effects.{mod_name}: {e}")
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith('effect_'):
                effect_id = name[len('effect_'):]
                # Look for variants function: variants_a001
                code = effect_id.split('_')[0]  # e.g. 'a001'
                variants_name = f'variants_{code}'
                variants_fn = getattr(mod, variants_name, None)
                registry[effect_id] = (fn, variants_fn)
    return registry


def list_effect_ids():
    """Return sorted list of all effect IDs."""
    return sorted(discover_effects().keys())


def get_effect(effect_id):
    """Get (effect_fn, variants_fn) for a single effect ID."""
    registry = discover_effects()
    return registry.get(effect_id)
