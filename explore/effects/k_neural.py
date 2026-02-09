"""K-series: Neural / ML-inspired effects (K001-K009).

Random neural network waveshapers, autoencoders, echo state networks,
neural ODEs, weight interpolation, convnet filter banks, tiny RNNs,
overfit-then-corrupt, and random projections.
"""
import numpy as np
import numba


# ============================================================================
# Shared helpers
# ============================================================================

def _overlap_add(samples, chunk_size, hop_size, process_fn):
    """Process *samples* in overlapping Hann-windowed chunks via *process_fn*.

    *process_fn(chunk)* receives a float64 chunk of length *chunk_size*
    and must return an array of the same length.
    """
    n = len(samples)
    window = np.hanning(chunk_size).astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        seg_len = end - pos
        chunk = np.zeros(chunk_size, dtype=np.float64)
        chunk[:seg_len] = samples[pos:end]
        chunk *= window

        processed = process_fn(chunk)

        out[pos:end] += (processed[:seg_len] * window[:seg_len])
        norm[pos:end] += window[:seg_len] ** 2
        pos += hop_size

    # Normalise where we have overlap
    mask = norm > 1e-8
    out[mask] /= norm[mask]
    return out


# ---------------------------------------------------------------------------
# K001 -- Random Neural Network Waveshaper
# ---------------------------------------------------------------------------

def effect_k001_random_neural_waveshaper(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """3-layer MLP with random weights applied to overlapping chunks.

    h1 = tanh(W1 @ chunk + b1)
    h2 = tanh(W2 @ h1 + b2)
    y  = W3 @ h2 + b3
    """
    hidden_size = int(params.get('hidden_size', 32))
    chunk_size = int(params.get('chunk_size', 64))
    seed = int(params.get('seed', 42))
    hop_size = chunk_size // 2

    rng = np.random.default_rng(seed)
    # Xavier-style init for stability
    scale1 = np.sqrt(2.0 / (chunk_size + hidden_size))
    W1 = rng.standard_normal((hidden_size, chunk_size)) * scale1
    b1 = rng.standard_normal(hidden_size) * 0.01
    scale2 = np.sqrt(2.0 / (hidden_size + hidden_size))
    W2 = rng.standard_normal((hidden_size, hidden_size)) * scale2
    b2 = rng.standard_normal(hidden_size) * 0.01
    scale3 = np.sqrt(2.0 / (hidden_size + chunk_size))
    W3 = rng.standard_normal((chunk_size, hidden_size)) * scale3
    b3 = rng.standard_normal(chunk_size) * 0.01

    def process_chunk(chunk):
        h1 = np.tanh(W1 @ chunk + b1)
        h2 = np.tanh(W2 @ h1 + b2)
        y = W3 @ h2 + b3
        return y

    x = samples.astype(np.float64)
    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k001():
    return [
        {'hidden_size': 16, 'chunk_size': 32, 'seed': 1},      # tiny net, glitchy
        {'hidden_size': 32, 'chunk_size': 64, 'seed': 42},     # default
        {'hidden_size': 64, 'chunk_size': 64, 'seed': 7},      # wider hidden
        {'hidden_size': 128, 'chunk_size': 128, 'seed': 99},   # large, smoother
        {'hidden_size': 32, 'chunk_size': 256, 'seed': 13},    # long chunks
        {'hidden_size': 64, 'chunk_size': 32, 'seed': 55},     # wide net, tiny chunks
    ]


# ---------------------------------------------------------------------------
# K002 -- Tiny Autoencoder
# ---------------------------------------------------------------------------

def effect_k002_tiny_autoencoder(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Train a tiny autoencoder (encoder->latent->decoder) on the input via
    simple numpy gradient descent, then corrupt the latent and decode."""
    latent_dim = int(params.get('latent_dim', 8))
    chunk_size = int(params.get('chunk_size', 128))
    corruption_type = str(params.get('corruption_type', 'noise'))
    corruption_amount = float(params.get('corruption_amount', 0.5))
    training_epochs = int(params.get('training_epochs', 100))
    hop_size = chunk_size // 2

    x = samples.astype(np.float64)

    # Collect training chunks
    chunks = []
    pos = 0
    window = np.hanning(chunk_size).astype(np.float64)
    while pos + chunk_size <= len(x):
        c = x[pos:pos + chunk_size] * window
        chunks.append(c)
        pos += hop_size
    if len(chunks) == 0:
        # Signal too short -- just return it
        return samples.copy()

    data = np.array(chunks)  # (num_chunks, chunk_size)

    # Init weights -- encoder: chunk_size -> latent_dim, decoder: latent_dim -> chunk_size
    rng = np.random.default_rng(7)
    scale_enc = np.sqrt(2.0 / (chunk_size + latent_dim))
    W_enc = rng.standard_normal((latent_dim, chunk_size)) * scale_enc
    b_enc = np.zeros(latent_dim)
    scale_dec = np.sqrt(2.0 / (latent_dim + chunk_size))
    W_dec = rng.standard_normal((chunk_size, latent_dim)) * scale_dec
    b_dec = np.zeros(chunk_size)

    lr = 0.001
    num_chunks = data.shape[0]

    # Training loop (batch gradient descent)
    for epoch in range(training_epochs):
        # Forward
        Z = np.tanh(data @ W_enc.T + b_enc)          # (N, latent_dim)
        recon = Z @ W_dec.T + b_dec                   # (N, chunk_size)
        error = recon - data                          # (N, chunk_size)

        # Backward
        # d_loss/d_recon = 2*error/N (MSE gradient)
        d_recon = (2.0 / num_chunks) * error          # (N, chunk_size)
        # Decoder grads
        dW_dec = d_recon.T @ Z                        # (chunk_size, latent_dim)
        db_dec = d_recon.sum(axis=0)
        # Encoder grads
        dZ = d_recon @ W_dec                          # (N, latent_dim)
        dZ_pre = dZ * (1.0 - Z ** 2)                 # tanh derivative
        dW_enc = dZ_pre.T @ data                      # (latent_dim, chunk_size)
        db_enc = dZ_pre.sum(axis=0)

        W_dec -= lr * dW_dec
        b_dec -= lr * db_dec
        W_enc -= lr * dW_enc
        b_enc -= lr * db_enc

    # Inference with corruption
    corrupt_rng = np.random.default_rng(42)

    def process_chunk(chunk):
        z = np.tanh(W_enc @ chunk + b_enc)
        # Corrupt latent
        if corruption_type == 'noise':
            z = z + corrupt_rng.standard_normal(latent_dim) * corruption_amount
        elif corruption_type == 'dropout':
            mask = corrupt_rng.random(latent_dim) > corruption_amount
            z = z * mask
        elif corruption_type == 'scale':
            z = z * corruption_amount
        recon = W_dec @ z + b_dec
        return recon

    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k002():
    return [
        {'latent_dim': 4, 'chunk_size': 64, 'corruption_type': 'noise',
         'corruption_amount': 0.3, 'training_epochs': 100},
        {'latent_dim': 8, 'chunk_size': 128, 'corruption_type': 'noise',
         'corruption_amount': 0.5, 'training_epochs': 100},
        {'latent_dim': 16, 'chunk_size': 128, 'corruption_type': 'noise',
         'corruption_amount': 1.0, 'training_epochs': 200},
        {'latent_dim': 8, 'chunk_size': 128, 'corruption_type': 'dropout',
         'corruption_amount': 0.5, 'training_epochs': 150},
        {'latent_dim': 8, 'chunk_size': 128, 'corruption_type': 'scale',
         'corruption_amount': 2.0, 'training_epochs': 100},
        {'latent_dim': 32, 'chunk_size': 256, 'corruption_type': 'noise',
         'corruption_amount': 0.2, 'training_epochs': 300},
        {'latent_dim': 4, 'chunk_size': 64, 'corruption_type': 'dropout',
         'corruption_amount': 0.8, 'training_epochs': 50},
    ]


# ---------------------------------------------------------------------------
# K003 -- Echo State Network
# ---------------------------------------------------------------------------

def effect_k003_echo_state_network(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Reservoir computing: fixed random reservoir, linear readout trained
    via least squares on the input signal."""
    reservoir_size = int(params.get('reservoir_size', 100))
    spectral_radius = float(params.get('spectral_radius', 0.95))
    input_scaling = float(params.get('input_scaling', 0.5))
    leak_rate = float(params.get('leak_rate', 0.3))

    x = samples.astype(np.float64)
    n = len(x)

    rng = np.random.default_rng(42)

    # Input weights
    W_in = (rng.standard_normal((reservoir_size, 1)) * input_scaling)

    # Reservoir weights -- sparse, scaled to desired spectral radius
    W_res = rng.standard_normal((reservoir_size, reservoir_size)) * 0.1
    # Make it sparse: zero out ~90% of connections
    sparsity_mask = rng.random((reservoir_size, reservoir_size)) < 0.1
    W_res *= sparsity_mask
    # Scale to spectral radius
    eigvals = np.linalg.eigvals(W_res)
    max_eigval = np.max(np.abs(eigvals))
    if max_eigval > 1e-10:
        W_res *= spectral_radius / max_eigval

    # Run reservoir
    states = np.zeros((n, reservoir_size), dtype=np.float64)
    h = np.zeros(reservoir_size, dtype=np.float64)
    for i in range(n):
        u = x[i]
        h_new = np.tanh(W_in[:, 0] * u + W_res @ h)
        h = (1.0 - leak_rate) * h + leak_rate * h_new
        states[i] = h

    # Train linear readout: x_target = states @ W_out
    # Use lstsq to find W_out that maps reservoir states -> original signal
    W_out, _, _, _ = np.linalg.lstsq(states, x, rcond=None)

    # Generate output -- the reservoir dynamics transform the signal
    # The readout was trained on the original, so divergence = the effect
    out = states @ W_out

    return out.astype(np.float32)


def variants_k003():
    return [
        {'reservoir_size': 50, 'spectral_radius': 0.9, 'input_scaling': 0.3,
         'leak_rate': 0.2},
        {'reservoir_size': 100, 'spectral_radius': 0.95, 'input_scaling': 0.5,
         'leak_rate': 0.3},
        {'reservoir_size': 200, 'spectral_radius': 1.0, 'input_scaling': 1.0,
         'leak_rate': 0.5},
        {'reservoir_size': 100, 'spectral_radius': 1.1, 'input_scaling': 0.5,
         'leak_rate': 0.1},
        {'reservoir_size': 300, 'spectral_radius': 0.85, 'input_scaling': 0.2,
         'leak_rate': 0.8},
        {'reservoir_size': 500, 'spectral_radius': 1.2, 'input_scaling': 2.0,
         'leak_rate': 1.0},
    ]


# ---------------------------------------------------------------------------
# K004 -- Neural ODE
# ---------------------------------------------------------------------------

def effect_k004_neural_ode(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """dx/dt = f(x, t) where f is a small MLP.  Euler integration over each chunk.

    The state x is a chunk of audio; f maps it through a random MLP,
    and we integrate forward in time.
    """
    hidden_size = int(params.get('hidden_size', 32))
    dt = float(params.get('dt', 0.05))
    num_steps = int(params.get('num_steps', 10))
    seed = int(params.get('seed', 42))
    chunk_size = 64
    hop_size = chunk_size // 2

    rng = np.random.default_rng(seed)

    # Two-layer MLP for the dynamics function f
    scale1 = np.sqrt(2.0 / (chunk_size + hidden_size))
    W1 = rng.standard_normal((hidden_size, chunk_size)) * scale1
    b1 = rng.standard_normal(hidden_size) * 0.01
    scale2 = np.sqrt(2.0 / (hidden_size + chunk_size))
    W2 = rng.standard_normal((chunk_size, hidden_size)) * scale2
    b2 = rng.standard_normal(chunk_size) * 0.01

    def dynamics(state):
        """f(x) = W2 @ tanh(W1 @ x + b1) + b2"""
        h = np.tanh(W1 @ state + b1)
        return W2 @ h + b2

    def process_chunk(chunk):
        state = chunk.copy()
        for _ in range(num_steps):
            state = state + dt * dynamics(state)
        return state

    x = samples.astype(np.float64)
    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k004():
    return [
        {'hidden_size': 16, 'dt': 0.01, 'num_steps': 5, 'seed': 1},    # gentle drift
        {'hidden_size': 32, 'dt': 0.05, 'num_steps': 10, 'seed': 42},  # default
        {'hidden_size': 64, 'dt': 0.02, 'num_steps': 20, 'seed': 7},   # many small steps
        {'hidden_size': 32, 'dt': 0.1, 'num_steps': 5, 'seed': 99},    # large steps, chaotic
        {'hidden_size': 16, 'dt': 0.05, 'num_steps': 50, 'seed': 13},  # long integration
        {'hidden_size': 64, 'dt': 0.1, 'num_steps': 30, 'seed': 55},   # extreme evolution
    ]


# ---------------------------------------------------------------------------
# K005 -- Weight Space Interpolation
# ---------------------------------------------------------------------------

def effect_k005_weight_space_interpolation(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Two random MLPs; interpolate their weights with parameter alpha,
    then process audio through the blended network."""
    hidden_size = int(params.get('hidden_size', 32))
    num_layers = int(params.get('num_layers', 3))
    seed_a = int(params.get('seed_a', 42))
    seed_b = int(params.get('seed_b', 123))
    alpha = float(params.get('alpha', 0.5))
    chunk_size = 64
    hop_size = chunk_size // 2

    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)

    # Build layer specs for both networks
    weights_a = []
    biases_a = []
    weights_b = []
    biases_b = []

    for i in range(num_layers):
        if i == 0:
            fan_in, fan_out = chunk_size, hidden_size
        elif i == num_layers - 1:
            fan_in, fan_out = hidden_size, chunk_size
        else:
            fan_in, fan_out = hidden_size, hidden_size

        scale = np.sqrt(2.0 / (fan_in + fan_out))
        weights_a.append(rng_a.standard_normal((fan_out, fan_in)) * scale)
        biases_a.append(rng_a.standard_normal(fan_out) * 0.01)
        weights_b.append(rng_b.standard_normal((fan_out, fan_in)) * scale)
        biases_b.append(rng_b.standard_normal(fan_out) * 0.01)

    # Interpolate
    weights = []
    biases = []
    for i in range(num_layers):
        weights.append((1.0 - alpha) * weights_a[i] + alpha * weights_b[i])
        biases.append((1.0 - alpha) * biases_a[i] + alpha * biases_b[i])

    def process_chunk(chunk):
        h = chunk
        for i in range(num_layers):
            h = weights[i] @ h + biases[i]
            if i < num_layers - 1:
                h = np.tanh(h)
        return h

    x = samples.astype(np.float64)
    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k005():
    return [
        {'hidden_size': 16, 'num_layers': 2, 'seed_a': 1, 'seed_b': 2, 'alpha': 0.0},
        {'hidden_size': 32, 'num_layers': 3, 'seed_a': 42, 'seed_b': 123, 'alpha': 0.5},
        {'hidden_size': 32, 'num_layers': 3, 'seed_a': 42, 'seed_b': 123, 'alpha': 1.0},
        {'hidden_size': 64, 'num_layers': 4, 'seed_a': 10, 'seed_b': 20, 'alpha': 0.25},
        {'hidden_size': 32, 'num_layers': 2, 'seed_a': 7, 'seed_b': 77, 'alpha': 0.75},
        {'hidden_size': 64, 'num_layers': 4, 'seed_a': 5, 'seed_b': 500, 'alpha': 0.1},
    ]


# ---------------------------------------------------------------------------
# K006 -- 1D Convnet Filter Bank
# ---------------------------------------------------------------------------

def effect_k006_convnet_filter_bank(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """N random 1D kernels, conv1d + ReLU, sum the outputs."""
    num_kernels = int(params.get('num_kernels', 8))
    kernel_size = int(params.get('kernel_size', 16))
    seed = int(params.get('seed', 42))

    rng = np.random.default_rng(seed)
    x = samples.astype(np.float64)
    n = len(x)

    # Generate random kernels, normalised so each has unit energy
    kernels = rng.standard_normal((num_kernels, kernel_size))
    for k in range(num_kernels):
        norm = np.sqrt(np.sum(kernels[k] ** 2))
        if norm > 1e-10:
            kernels[k] /= norm

    out = np.zeros(n, dtype=np.float64)
    for k in range(num_kernels):
        # Full convolution then truncate to input length
        conv = np.convolve(x, kernels[k], mode='same')
        # ReLU
        conv = np.maximum(conv, 0.0)
        out += conv

    # Normalise to roughly match input level
    out /= num_kernels
    rms_in = np.sqrt(np.mean(x ** 2)) + 1e-10
    rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= rms_in / rms_out

    return out.astype(np.float32)


def variants_k006():
    return [
        {'num_kernels': 4, 'kernel_size': 3, 'seed': 1},     # few tiny kernels
        {'num_kernels': 8, 'kernel_size': 16, 'seed': 42},   # default
        {'num_kernels': 16, 'kernel_size': 8, 'seed': 7},    # many short kernels
        {'num_kernels': 8, 'kernel_size': 64, 'seed': 99},   # long kernels, resonant
        {'num_kernels': 32, 'kernel_size': 32, 'seed': 13},  # dense filter bank
        {'num_kernels': 4, 'kernel_size': 64, 'seed': 55},   # few long filters
    ]


# ---------------------------------------------------------------------------
# K007 -- Tiny RNN Sample Processor
# ---------------------------------------------------------------------------

@numba.njit
def _rnn_process(samples, w_ih, w_hh, b_h, w_ho, hidden_size):
    """Single-layer RNN applied sample by sample.

    h = tanh(w_ih * x + w_hh @ h_prev + b_h)
    y = w_ho @ h
    """
    n = len(samples)
    out = np.zeros(n, dtype=np.float64)
    h = np.zeros(hidden_size, dtype=np.float64)

    for i in range(n):
        x = samples[i]
        h_new = np.zeros(hidden_size, dtype=np.float64)
        for j in range(hidden_size):
            val = w_ih[j] * x + b_h[j]
            for k in range(hidden_size):
                val += w_hh[j, k] * h[k]
            h_new[j] = np.tanh(val)
        h = h_new

        y = 0.0
        for j in range(hidden_size):
            y += w_ho[j] * h[j]
        out[i] = y

    return out


def effect_k007_tiny_rnn(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Single RNN cell applied sample-by-sample via numba."""
    hidden_size = int(params.get('hidden_size', 8))
    seed = int(params.get('seed', 42))

    rng = np.random.default_rng(seed)

    # Input->hidden weights (1 input)
    scale_ih = np.sqrt(2.0 / (1 + hidden_size))
    w_ih = rng.standard_normal(hidden_size) * scale_ih
    # Hidden->hidden weights
    scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
    w_hh = rng.standard_normal((hidden_size, hidden_size)) * scale_hh
    # Bias
    b_h = rng.standard_normal(hidden_size) * 0.01
    # Hidden->output weights (1 output)
    scale_ho = np.sqrt(2.0 / (hidden_size + 1))
    w_ho = rng.standard_normal(hidden_size) * scale_ho

    x = samples.astype(np.float64)
    out = _rnn_process(x, w_ih, w_hh, b_h, w_ho, hidden_size)

    # Normalise output RMS to match input
    rms_in = np.sqrt(np.mean(x ** 2)) + 1e-10
    rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= rms_in / rms_out

    return out.astype(np.float32)


def variants_k007():
    return [
        {'hidden_size': 4, 'seed': 1},     # tiny, minimal nonlinearity
        {'hidden_size': 8, 'seed': 42},    # default
        {'hidden_size': 16, 'seed': 7},    # wider, more complex dynamics
        {'hidden_size': 32, 'seed': 99},   # large, rich timbre
        {'hidden_size': 8, 'seed': 13},    # same size, different character
        {'hidden_size': 4, 'seed': 200},   # tiny, alternate personality
    ]


# ---------------------------------------------------------------------------
# K008 -- Overfit-Then-Corrupt
# ---------------------------------------------------------------------------

def effect_k008_overfit_then_corrupt(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Train an MLP to reconstruct audio chunks (overfit), then corrupt weights
    and use the corrupted network to process audio.

    Corruption modes: weight_noise, quantize, prune.
    Pure numpy gradient descent -- no external ML libraries.
    """
    hidden_size = int(params.get('hidden_size', 32))
    chunk_size = int(params.get('chunk_size', 64))
    corruption_type = str(params.get('corruption_type', 'weight_noise'))
    corruption_amount = float(params.get('corruption_amount', 0.5))
    train_epochs = int(params.get('train_epochs', 100))
    hop_size = chunk_size // 2

    x = samples.astype(np.float64)

    # Collect training chunks
    window = np.hanning(chunk_size).astype(np.float64)
    chunks = []
    pos = 0
    while pos + chunk_size <= len(x):
        c = x[pos:pos + chunk_size] * window
        chunks.append(c)
        pos += hop_size
    if len(chunks) == 0:
        return samples.copy()

    data = np.array(chunks)  # (N, chunk_size)

    # Init 2-layer MLP: chunk_size -> hidden -> chunk_size
    rng = np.random.default_rng(7)
    scale1 = np.sqrt(2.0 / (chunk_size + hidden_size))
    W1 = rng.standard_normal((hidden_size, chunk_size)) * scale1
    b1 = np.zeros(hidden_size)
    scale2 = np.sqrt(2.0 / (hidden_size + chunk_size))
    W2 = rng.standard_normal((chunk_size, hidden_size)) * scale2
    b2 = np.zeros(chunk_size)

    lr = 0.001
    num_chunks = data.shape[0]

    # Train (overfit)
    for epoch in range(train_epochs):
        # Forward: H = tanh(data @ W1.T + b1), recon = H @ W2.T + b2
        H = np.tanh(data @ W1.T + b1)        # (N, hidden)
        recon = H @ W2.T + b2                 # (N, chunk_size)
        error = recon - data                  # (N, chunk_size)

        # Backward
        d_recon = (2.0 / num_chunks) * error
        dW2 = d_recon.T @ H                  # (chunk_size, hidden)
        db2 = d_recon.sum(axis=0)
        dH = d_recon @ W2                    # (N, hidden)
        dH_pre = dH * (1.0 - H ** 2)        # tanh derivative
        dW1 = dH_pre.T @ data               # (hidden, chunk_size)
        db1 = dH_pre.sum(axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # Corrupt weights
    corrupt_rng = np.random.default_rng(42)
    if corruption_type == 'weight_noise':
        W1 = W1 + corrupt_rng.standard_normal(W1.shape) * corruption_amount * np.std(W1)
        W2 = W2 + corrupt_rng.standard_normal(W2.shape) * corruption_amount * np.std(W2)
        b1 = b1 + corrupt_rng.standard_normal(b1.shape) * corruption_amount * np.std(b1 + 1e-10)
        b2 = b2 + corrupt_rng.standard_normal(b2.shape) * corruption_amount * np.std(b2 + 1e-10)
    elif corruption_type == 'quantize':
        # Quantize weights to fewer levels
        num_levels = max(2, int(16 / corruption_amount))
        for arr in [W1, W2, b1, b2]:
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-10:
                arr_norm = (arr - mn) / (mx - mn)
                arr_q = np.round(arr_norm * num_levels) / num_levels
                arr[:] = arr_q * (mx - mn) + mn
    elif corruption_type == 'prune':
        # Zero out a fraction of weights
        for arr in [W1, W2]:
            mask = corrupt_rng.random(arr.shape) > corruption_amount
            arr *= mask

    def process_chunk(chunk):
        h = np.tanh(W1 @ chunk + b1)
        return W2 @ h + b2

    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k008():
    return [
        {'hidden_size': 16, 'chunk_size': 64, 'corruption_type': 'weight_noise',
         'corruption_amount': 0.3, 'train_epochs': 100},
        {'hidden_size': 32, 'chunk_size': 64, 'corruption_type': 'weight_noise',
         'corruption_amount': 0.5, 'train_epochs': 100},
        {'hidden_size': 32, 'chunk_size': 128, 'corruption_type': 'weight_noise',
         'corruption_amount': 1.5, 'train_epochs': 200},
        {'hidden_size': 32, 'chunk_size': 64, 'corruption_type': 'quantize',
         'corruption_amount': 0.5, 'train_epochs': 100},
        {'hidden_size': 32, 'chunk_size': 64, 'corruption_type': 'prune',
         'corruption_amount': 0.7, 'train_epochs': 100},
        {'hidden_size': 64, 'chunk_size': 256, 'corruption_type': 'quantize',
         'corruption_amount': 2.0, 'train_epochs': 150},
        {'hidden_size': 16, 'chunk_size': 64, 'corruption_type': 'prune',
         'corruption_amount': 0.9, 'train_epochs': 50},
    ]


# ---------------------------------------------------------------------------
# K009 -- Random Projection
# ---------------------------------------------------------------------------

def effect_k009_random_projection(samples: np.ndarray, sr: int, **params) -> np.ndarray:
    """Project chunks to a lower-dimensional space and back.

    chunk -> A @ chunk  (down to bottleneck_dim)
    latent -> B @ latent (back up to chunk_size)
    Information loss in the bottleneck creates the effect.
    """
    chunk_size = int(params.get('chunk_size', 64))
    bottleneck_dim = int(params.get('bottleneck_dim', 8))
    seed = int(params.get('seed', 42))
    hop_size = chunk_size // 2

    rng = np.random.default_rng(seed)

    # Random projection matrices (orthogonalised for stability)
    A_raw = rng.standard_normal((bottleneck_dim, chunk_size))
    # Orthogonalise rows via QR
    Q, _ = np.linalg.qr(A_raw.T)
    A = Q[:, :bottleneck_dim].T  # (bottleneck_dim, chunk_size)

    B_raw = rng.standard_normal((chunk_size, bottleneck_dim))
    Q2, _ = np.linalg.qr(B_raw)
    B = Q2[:, :bottleneck_dim]   # (chunk_size, bottleneck_dim)

    def process_chunk(chunk):
        latent = A @ chunk
        return B @ latent

    x = samples.astype(np.float64)
    return _overlap_add(x, chunk_size, hop_size, process_chunk).astype(np.float32)


def variants_k009():
    return [
        {'chunk_size': 32, 'bottleneck_dim': 2, 'seed': 1},    # extreme compression
        {'chunk_size': 64, 'bottleneck_dim': 8, 'seed': 42},   # default
        {'chunk_size': 64, 'bottleneck_dim': 4, 'seed': 7},    # more lossy
        {'chunk_size': 128, 'bottleneck_dim': 16, 'seed': 99}, # moderate, longer chunks
        {'chunk_size': 256, 'bottleneck_dim': 32, 'seed': 13}, # large chunks, mild
        {'chunk_size': 64, 'bottleneck_dim': 2, 'seed': 55},   # heavy bottleneck
    ]
