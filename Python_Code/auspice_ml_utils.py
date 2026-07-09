"""
auspice_ml_utils.py — Shared utilities for the aTTC surrogate ML pipeline.

Provides data loading (.mat files from MATLAB or Python), feature extraction,
model building, and loss function lookup used by split_structs, _train,
_test, and _master scripts.
"""

import json
import numpy as np
from pathlib import Path

##### Paths #####
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
#                           DATA  LOADING
# ============================================================================

def load_mat_v73(path):
    """Load a MATLAB v7.3 (HDF5) .mat file and return a dict of numpy arrays."""
    import h5py
    out = {}
    with h5py.File(str(path), "r") as f:
        # Find the results struct — may be named 'results', 'results_2d', etc.
        # Also handles flat layout (fields at top level, no 'results' key).
        result_key = None
        for k in f.keys():
            if k.startswith("results") or k == "results":
                result_key = k
                break

        if result_key is not None:
            root = f[result_key]
        else:
            # Flat layout: fields at top level
            root = f

        for key in root.keys():
            if key == '#refs#':
                continue
            if key == 'params':
                continue  # handle below
            try:
                out[key] = np.array(root[key])
            except Exception:
                pass  # skip non-array fields

        # Load scalar params
        params = {}
        if 'params' in root:
            p = root["params"]
            for pk in p.keys():
                try:
                    val = p[pk][()]
                    if val.ndim == 0 or val.size == 1:
                        params[pk] = float(val.flat[0])
                    else:
                        params[pk] = np.array(val).squeeze()
                except Exception:
                    pass
        out["params"] = params
    return out


def _convert_param_value(val):
    """Convert a param value loaded from .mat to a Python scalar or array."""
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        return val.decode()
    if isinstance(val, np.ndarray):
        # Check for string arrays first
        if val.dtype.kind in ('U', 'S', 'O'):
            s = str(val.flat[0])
            if isinstance(val.flat[0], bytes):
                s = val.flat[0].decode()
            return s
        if val.size == 1:
            return float(val.flat[0])
        return val.squeeze()
    if np.isscalar(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val
    return val


def load_mat_scipy(path):
    """Load a MATLAB v5 .mat file via scipy."""
    import scipy.io as sio
    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # Check for a 'results' struct wrapper, fall back to flat layout
    if "results" in raw:
        results = raw["results"]
    else:
        # Flat layout: use the raw dict directly
        # Remove meta keys
        results = None
        flat_keys = [k for k in raw.keys() if not k.startswith("__")]
        if flat_keys:
            results = raw

    if results is None:
        raise KeyError(f"No data found in {path}")

    # If results is a struct object, extract attributes
    if not isinstance(results, dict):
        out = {}
        for attr in dir(results):
            if attr.startswith("_"):
                continue
            val = getattr(results, attr)
            if isinstance(val, np.ndarray):
                out[attr] = val
        # params
        p = results.params
        params = {}
        for attr in dir(p):
            if attr.startswith("_"):
                continue
            val = getattr(p, attr)
            params[attr] = _convert_param_value(val)
        out["params"] = params
    else:
        # Flat dict layout
        out = {}
        for k, v in results.items():
            if k.startswith("__"):
                continue
            if k == "params":
                continue
            if isinstance(v, np.ndarray):
                out[k] = v
        # params
        if "params" in results:
            p = results["params"]
            params = {}
            if isinstance(p, np.ndarray) and p.dtype.names is not None:
                # structured array
                for name in p.dtype.names:
                    val = p[name].flat[0]
                    params[name] = _convert_param_value(val)
            elif hasattr(p, '__dict__') or hasattr(p, '_fieldnames'):
                for attr in dir(p):
                    if attr.startswith("_"):
                        continue
                    val = getattr(p, attr)
                    params[attr] = _convert_param_value(val)
            out["params"] = params
        else:
            out["params"] = {}
    return out


def load_mat(path):
    """Auto-detect .mat version and load."""
    path = str(path)
    try:
        return load_mat_v73(path)
    except Exception:
        return load_mat_scipy(path)


def load_keras_model(path):
    """Load a Keras .h5 model, working around h5py 3.x / TF 2.0 incompatibility.

    h5py >= 3 returns HDF5 string attributes as Python ``str``, but
    TensorFlow 2.0's ``load_model`` and ``load_weights`` call
    ``.decode('utf-8')`` on them, raising ``AttributeError``.

    This wrapper temporarily patches h5py so that string attributes are
    returned as ``bytes``, then restores the original behaviour.
    """
    import h5py
    import tensorflow as tf

    # Monkey-patch h5py so str attrs come back as bytes (TF 2.0 compat).
    # h5py 3.x returns str for scalars and np.str_ arrays for lists;
    # TF 2.0 expects bytes everywhere so it can call .decode().
    _orig = h5py.AttributeManager.__getitem__

    def _bytes_getitem(self, name):
        val = _orig(self, name)
        if isinstance(val, str):
            return val.encode("utf-8")
        if isinstance(val, np.ndarray) and val.dtype.kind in ("U", "O"):
            return np.array([v.encode("utf-8") if isinstance(v, str) else v
                             for v in val.flat], dtype=object).reshape(val.shape)
        return val

    h5py.AttributeManager.__getitem__ = _bytes_getitem
    try:
        # Ensure SoftCap custom layer is registered before deserialisation
        _get_soft_cap_layer_class()
        return tf.keras.models.load_model(str(path))
    finally:
        h5py.AttributeManager.__getitem__ = _orig


def normalize_results(data):
    """Ensure arrays are in canonical Python layout (time axis first).

    MATLAB v7.3 (HDF5) stores arrays transposed:
      x:    (state_dim, num_agents, N) → should be (N, num_agents, state_dim)
      attc: (num_pairs, N)             → should be (N, num_pairs)
      2-D:  (cols, rows)               → should be (rows, cols)

    This function detects the MATLAB layout and transposes as needed.
    Returns a new dict (does not modify the original).
    """
    out = dict(data)  # shallow copy

    # Convert MATLAB char arrays in params to Python strings
    if "params" in out and isinstance(out["params"], dict):
        p = out["params"]
        for k, v in p.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S', 'u', 'i', 'f'):
                # MATLAB char arrays may come as numeric arrays of ASCII codes
                try:
                    if v.dtype.kind in ('i', 'u', 'f') and v.ndim == 1 and v.size > 0:
                        if all(32 <= c < 127 for c in v.flat):
                            p[k] = ''.join(chr(int(c)) for c in v.flat)
                    elif v.dtype.kind in ('U', 'S'):
                        p[k] = str(v.flat[0]) if v.size == 1 else str(v)
                except (ValueError, TypeError):
                    pass

    ##### x: 3-D state array #####
    if "x" in out:
        x = out["x"]
        if x.ndim == 3 and x.shape[0] <= 6:
            # HDF5 layout: (sd, na, N) → (N, na, sd)
            out["x"] = np.transpose(x, (2, 1, 0))

    ##### attc: 2-D array #####
    if "attc" in out:
        a = out["attc"]
        if a.ndim == 2 and a.shape[0] < a.shape[1]:
            # HDF5 layout: (num_pairs, N) → (N, num_pairs)
            out["attc"] = a.T
        elif a.ndim == 1:
            out["attc"] = a[:, np.newaxis]

    ##### Other 2-D time-series: ttc, sttc1, sttc2, dist, h, etc. #####
    for key in ("ttc", "sttc1", "sttc2", "ttc2", "sttc2_1sig", "sttc2_2sig",
                "dist", "h", "sigma_pos", "r_eff"):
        if key in out:
            arr = out[key]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                out[key] = arr.T

    ##### u: 3-D control array #####
    if "u" in out:
        u = out["u"]
        if u.ndim == 3 and u.shape[0] <= 2:
            # HDF5 layout: (cd, na, N) → (N, na, cd)
            out["u"] = np.transpose(u, (2, 1, 0))

    return out


# ============================================================================
#                      FEATURE  EXTRACTION
# ============================================================================

def state_to_pos_vel(x, dim):
    """Extract position and Cartesian velocity from state array.

    Parameters
    ----------
    x : ndarray, shape (N, state_dim)
        State vectors for one agent across N timesteps.
    dim : int (2 or 3)

    Returns
    -------
    pos : ndarray (N, dim)
    vel : ndarray (N, dim)
    """
    if dim == 2:
        px, py = x[:, 0], x[:, 1]
        theta  = x[:, 2]
        v      = x[:, 3]
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        return np.column_stack([px, py]), np.column_stack([vx, vy])
    else:
        px, py, pz = x[:, 0], x[:, 1], x[:, 2]
        psi   = x[:, 3]
        gamma = x[:, 4]
        v     = x[:, 5]
        vx = v * np.cos(gamma) * np.cos(psi)
        vy = v * np.cos(gamma) * np.sin(psi)
        vz = v * np.sin(gamma)
        return np.column_stack([px, py, pz]), np.column_stack([vx, vy, vz])


def extract_pair_features(data, pair_idx=0, attc_cap=None, attc_filter=None):
    """Extract (delta_p, delta_v, attc) for a single agent pair.

    Parameters
    ----------
    data : dict  — must have 'x', 'attc', 'params' (already normalized layout)
    pair_idx : int  (0-based)
    attc_cap : float or None
        If not None, replace inf aTTC values with this cap (seconds) instead
        of discarding them.  NaN and non-positive values are still dropped.
    attc_filter : float or None
        If not None, discard all samples with aTTC > attc_filter (seconds).
        Applied after attc_cap, so capped values can also be filtered out.

    Returns
    -------
    features : ndarray (N_valid, 2*dim)   — [delta_px, delta_py, (delta_pz,) delta_vx, delta_vy, (delta_vz)]
    targets  : ndarray (N_valid,)          — finite positive aTTC values
    """
    params = data["params"]
    dim = int(params["dim"])
    na  = int(params["num_agents"])
    x   = data["x"]       # (N, na, sd)  — already normalized
    attc = data["attc"]   # (N, num_pairs)

    N = x.shape[0]

    # Map pair_idx → (agent_i, agent_j)
    pair = 0
    agent_i, agent_j = 0, 1
    for i in range(na):
        for j in range(i + 1, na):
            if pair == pair_idx:
                agent_i, agent_j = i, j
            pair += 1

    # Compute relative position & velocity
    pos_i, vel_i = state_to_pos_vel(x[:, agent_i, :], dim)
    pos_j, vel_j = state_to_pos_vel(x[:, agent_j, :], dim)

    delta_p = pos_i - pos_j   # (N, dim)  i-j convention (matches MATLAB)
    delta_v = vel_i - vel_j   # (N, dim)  i-j convention (matches MATLAB)

    features = np.hstack([delta_p, delta_v])  # (N, 2*dim)
    targets  = attc[:, pair_idx].copy()       # (N,)

    # Clamp targets to min(target, cap) if requested.
    #   - Inf values are replaced by cap (they're "at least cap").
    #   - Finite values above cap are clamped down to cap.
    # This ensures the training target distribution has no mass above cap,
    # which is consistent with the SoftCap output layer in build_model.
    n_inf_capped = 0
    n_fin_capped = 0
    if attc_cap is not None:
        inf_mask = np.isinf(targets) & (targets > 0)
        n_inf_capped = int(inf_mask.sum())
        targets[inf_mask] = attc_cap
        fin_above_mask = np.isfinite(targets) & (targets > attc_cap)
        n_fin_capped = int(fin_above_mask.sum())
        targets[fin_above_mask] = attc_cap

    # Filter out remaining NaN / non-positive (keep finite, including capped)
    valid = np.isfinite(targets) & ~np.isnan(targets) & (targets > 0)

    # Filter out samples above attc_filter threshold
    n_filtered = 0
    if attc_filter is not None:
        above = valid & (targets > attc_filter)
        n_filtered = int(above.sum())
        valid = valid & (targets <= attc_filter)

    cap_str = ""
    if attc_cap is not None:
        cap_str = (f", {n_inf_capped} Inf→cap, "
                   f"{n_fin_capped} finite clamped to cap={attc_cap:.0f}s")
    filt_str = ""
    if attc_filter is not None:
        filt_str = f", {n_filtered} filtered (>{attc_filter:.0f}s)"
    print(f"  Pair ({agent_i+1},{agent_j+1}): {N} timesteps, "
          f"{valid.sum()} valid aTTC samples "
          f"({100*valid.mean():.1f}%){cap_str}{filt_str}")

    return features[valid].astype(np.float32), targets[valid].astype(np.float32)


def parse_pair_idx(pair_idx_arg, num_agents):
    """Parse a pair_idx argument into a list of 0-based pair indices.

    Parameters
    ----------
    pair_idx_arg : int, str, or list
        Accepted formats:
          - int: single pair index  (e.g. 0)
          - "all": all pairs
          - comma-separated ints:   (e.g. "0,2,5")
          - list of ints:           (e.g. [0, 2, 5])
    num_agents : int
        Number of agents (used to compute total number of pairs).

    Returns
    -------
    list of int — sorted, validated pair indices.
    """
    num_pairs = num_agents * (num_agents - 1) // 2

    if isinstance(pair_idx_arg, (list, tuple)):
        indices = [int(x) for x in pair_idx_arg]
    elif isinstance(pair_idx_arg, int):
        indices = [pair_idx_arg]
    else:
        s = str(pair_idx_arg).strip().lower()
        if s == "all":
            return list(range(num_pairs))
        indices = [int(x.strip()) for x in s.split(",")]

    # Validate
    for idx in indices:
        if idx < 0 or idx >= num_pairs:
            raise ValueError(
                f"pair_idx {idx} out of range [0, {num_pairs - 1}] "
                f"for {num_agents} agents ({num_pairs} pairs)")
    return sorted(set(indices))


def extract_multi_pair_features(data, pair_indices, attc_cap=None,
                                attc_filter=None):
    """Extract features for one or more pairs, concatenated.

    Parameters
    ----------
    data : dict — must have 'x', 'attc', 'params' (already normalized layout)
    pair_indices : list of int — 0-based pair indices
    attc_cap : float or None
    attc_filter : float or None

    Returns
    -------
    features : ndarray (N_total, 2*dim)
    targets  : ndarray (N_total,)
    """
    all_feat, all_tgt = [], []
    for p in pair_indices:
        feat, tgt = extract_pair_features(data, p, attc_cap=attc_cap,
                                          attc_filter=attc_filter)
        all_feat.append(feat)
        all_tgt.append(tgt)

    return np.concatenate(all_feat, axis=0), np.concatenate(all_tgt, axis=0)


def extract_all_pair_features(data, attc_cap=None, attc_filter=None):
    """Extract features for ALL pairs, concatenated.

    Returns
    -------
    features : ndarray (N_total, 2*dim)
    targets  : ndarray (N_total,)
    """
    params = data["params"]
    na = int(params["num_agents"])
    num_pairs = na * (na - 1) // 2

    return extract_multi_pair_features(
        data, list(range(num_pairs)), attc_cap=attc_cap,
        attc_filter=attc_filter)


# ============================================================================
#                            MODEL
# ============================================================================

_SoftCapLayerClass = None

def _get_soft_cap_layer_class():
    """Lazy-build the SoftCap Keras Layer class on first use.

    Registered with Keras's custom-object serialization so that
    save_model/load_model round-trip without needing custom_objects=.
    """
    global _SoftCapLayerClass
    if _SoftCapLayerClass is not None:
        return _SoftCapLayerClass

    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable(package="attc_surrogate",
                                                 name="SoftCap")
    class SoftCap(layers.Layer):
        """Smooth soft-min cap:
            pred = cap - softplus(cap - raw)

        Properties:
          * pred -> raw when raw << cap (pass-through in the low-aTTC zone)
          * pred -> cap asymptotically as raw -> infinity (smooth saturation)
          * Gradient is strong at low aTTC, vanishes as prediction saturates.
        """
        def __init__(self, cap, **kwargs):
            super().__init__(**kwargs)
            self.cap = float(cap)

        def call(self, raw):
            return self.cap - tf.keras.activations.softplus(self.cap - raw)

        def get_config(self):
            base = super().get_config()
            base.update({"cap": self.cap})
            return base

    _SoftCapLayerClass = SoftCap
    return SoftCap


def _soft_cap_layer(cap):
    """Factory for the soft-cap output layer (lazy Keras import)."""
    return _get_soft_cap_layer_class()(cap=cap, name="soft_cap")


def build_model(input_dim, hidden_sizes=(64, 64, 32), film=False,
                cond_hidden=(16, 16), attc_cap=None):
    """Build a dense regression network, optionally with FiLM conditioning.

    Parameters
    ----------
    input_dim : int
        Dimension of the state features (2*dim).
    hidden_sizes : tuple of int
        Hidden layer sizes for the main pathway.
    film : bool
        If True, build a FiLM-conditioned model with a second input for v_max.
    cond_hidden : tuple of int
        Hidden layer sizes for the conditioning MLP (only used if film=True).
    attc_cap : float or None
        If not None, apply a smooth soft-min cap at the output so that
        predictions saturate at ``attc_cap`` seconds.  Output formula:
            pred = cap - softplus(cap - softplus(a4))
        If None, output is plain softplus (original behaviour, unchanged).

    Returns
    -------
    keras.Model
        Sequential model (film=False) or Functional API model (film=True).
    """
    if film:
        return _build_film_model(input_dim, hidden_sizes, cond_hidden,
                                 attc_cap=attc_cap)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="attc_surrogate")
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for h in hidden_sizes:
        model.add(layers.Dense(h, activation="relu"))
    # Final softplus produces unbounded positive prediction
    model.add(layers.Dense(1, activation="softplus", name="output_raw"))
    # Optional soft-cap
    if attc_cap is not None:
        model.add(_soft_cap_layer(attc_cap))
    return model


def _build_film_model(input_dim, hidden_sizes=(64, 64, 32),
                      cond_hidden=(16, 16), attc_cap=None):
    """Build a FiLM-conditioned model (Keras Functional API).

    Two inputs:
        state_features : (batch, input_dim) — delta_p, delta_v
        v_max_input    : (batch, 1)         — conditioning variable

    The conditioning MLP maps v_max to per-layer (gamma, beta) pairs that
    modulate hidden activations before ReLU:
        a_mod = gamma * a + beta;  h = ReLU(a_mod)

    Gamma biases initialised to 1, beta biases to 0 (identity at init).
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # --- Inputs ---
    state_input = layers.Input(shape=(input_dim,), name="state_features")
    cond_input  = layers.Input(shape=(1,), name="v_max_input")

    # --- Conditioning pathway ---
    # Small MLP: v_max → gamma/beta for all hidden layers
    total_film_params = sum(h * 2 for h in hidden_sizes)  # gamma + beta per layer
    c = cond_input
    for i, ch in enumerate(cond_hidden):
        c = layers.Dense(ch, activation="relu",
                         name=f"cond_dense_{i+1}")(c)
    # Final layer produces all gammas and betas concatenated
    # Initialise so gamma ≈ 1, beta ≈ 0 at start
    film_params = layers.Dense(
        total_film_params, activation="linear",
        name="cond_film_output",
        bias_initializer=_film_bias_init(hidden_sizes)
    )(c)

    # Split into per-layer gamma/beta
    gammas, betas = [], []
    offset = 0
    for h in hidden_sizes:
        gammas.append(layers.Lambda(
            lambda x, o=offset, s=h: x[:, o:o+s],
            name=f"gamma_{len(gammas)+1}")(film_params))
        offset += h
        betas.append(layers.Lambda(
            lambda x, o=offset, s=h: x[:, o:o+s],
            name=f"beta_{len(betas)+1}")(film_params))
        offset += h

    # --- Main pathway with FiLM ---
    x = state_input
    for i, h in enumerate(hidden_sizes):
        # Dense (no activation)
        a = layers.Dense(h, activation=None,
                         name=f"main_dense_{i+1}")(x)
        # FiLM: gamma * a + beta
        a_mod = layers.Multiply(name=f"film_mul_{i+1}")([gammas[i], a])
        a_mod = layers.Add(name=f"film_add_{i+1}")([a_mod, betas[i]])
        # ReLU
        x = layers.Activation("relu", name=f"relu_{i+1}")(a_mod)

    # Output layer (softplus, no FiLM)
    output = layers.Dense(1, activation="softplus", name="output_raw")(x)

    # Optional soft-cap: pred = cap - softplus(cap - softplus(a4))
    if attc_cap is not None:
        output = _soft_cap_layer(attc_cap)(output)

    model = keras.Model(inputs=[state_input, cond_input], outputs=output,
                        name="attc_surrogate_film")
    return model


def _film_bias_init(hidden_sizes):
    """Custom bias initialiser for FiLM output layer.

    Sets gamma biases to 1.0 and beta biases to 0.0 so the model starts
    as an identity modulation.
    """
    import tensorflow as tf

    bias_values = []
    for h in hidden_sizes:
        bias_values.extend([1.0] * h)  # gamma = 1
        bias_values.extend([0.0] * h)  # beta = 0

    return tf.constant_initializer(bias_values)


def load_multi_vmax_data(mat_files, pair_idx=0, attc_cap=None,
                         attc_filter=None):
    """Load and combine data from multiple .mat files with different v_max.

    Each file should contain results.params.v_max_attc (or v_max).
    Returns features, targets, and a v_max array aligned per sample.

    Parameters
    ----------
    mat_files : list of str/Path
        Paths to .mat files, each from a different v_max run.
    pair_idx : int or str
        Pair index (0-based int) or 'all'.
    attc_cap : float or None
        Cap infinite aTTC values at this threshold.
    attc_filter : float or None
        Discard samples with aTTC > this threshold.

    Returns
    -------
    features  : ndarray (N_total, 2*dim)
    targets   : ndarray (N_total,)
    v_max_arr : ndarray (N_total,)
    dim       : int (2 or 3)
    per_file_sizes : list of int — number of valid samples per file
    """
    all_feat, all_tgt, all_vmax = [], [], []
    per_file_sizes = []
    dim = None

    for path in mat_files:
        data = load_mat(str(path))
        data = normalize_results(data)
        params = data["params"]

        d = int(params["dim"])
        if dim is None:
            dim = d
        else:
            assert d == dim, f"Dimension mismatch: expected {dim}, got {d}"

        # Get v_max for this file
        v_max = params.get("v_max_attc", params.get("v_max", None))
        if v_max is None:
            raise ValueError(f"No v_max found in {path}")
        v_max = float(v_max)

        # Extract features
        if isinstance(pair_idx, str) and pair_idx == 'all':
            feat, tgt = extract_all_pair_features(data, attc_cap=attc_cap,
                                                  attc_filter=attc_filter)
        elif isinstance(pair_idx, list):
            feat, tgt = extract_multi_pair_features(data, pair_idx,
                                                    attc_cap=attc_cap,
                                                    attc_filter=attc_filter)
        else:
            feat, tgt = extract_pair_features(data, int(pair_idx),
                                              attc_cap=attc_cap,
                                              attc_filter=attc_filter)

        n = feat.shape[0]
        all_feat.append(feat)
        all_tgt.append(tgt)
        all_vmax.append(np.full(n, v_max, dtype=np.float32))
        per_file_sizes.append(n)
        print(f"  v_max={v_max:.3f}: {n} samples")

    features  = np.concatenate(all_feat, axis=0)
    targets   = np.concatenate(all_tgt, axis=0)
    v_max_arr = np.concatenate(all_vmax, axis=0)

    print(f"  Total: {features.shape[0]} samples, dim={dim}, "
          f"v_max range [{v_max_arr.min():.3f}, {v_max_arr.max():.3f}]")

    return features, targets, v_max_arr, dim, per_file_sizes


# ============================================================================
#                   OUTPUT-WEIGHTED SAMPLE WEIGHTS
# ============================================================================

def compute_ow_weights(features, targets, kde_bw='scott', clip_max=50.0,
                       kde_subsample=10000, mode='full'):
    """Compute output-weighted sample weights (Blanchard & Sapsis 2021).

    Weights are the likelihood ratio  w(x,y) = p_x(x) / p_y(y_true)
    where p_x is the input density and p_y is the output density,
    both estimated via Gaussian KDE.

    Samples with rare output values (tails of p_y) receive higher weight,
    biasing the loss toward safety-critical low-aTTC events.

    Parameters
    ----------
    features : ndarray (N, d)
        Input feature matrix (e.g. normalised delta_p, delta_v).
    targets : ndarray (N,)
        Output targets (e.g. finite aTTC values in seconds).
    kde_bw : str or float
        Bandwidth method for KDE: 'scott', 'silverman', or a scalar.
    clip_max : float
        Maximum allowed weight (after normalisation to mean=1) to prevent
        extreme outlier weights from destabilising training.  Default 50.
    kde_subsample : int
        Maximum number of samples used to *fit* each KDE.  The fitted KDE
        is then evaluated on all N samples.  Subsampling keeps the KDE
        fitting tractable for large datasets.  Default 10000.
    mode : str
        'full'  — w = p_x(x) / p_y(y) using marginal KDEs for p_x.
        'output_only' — w = 1 / p_y(y), assumes uniform input density.
                         Appropriate for uniformly-sampled time series.

    Returns
    -------
    weights : ndarray (N,)
        Per-sample weights with mean = 1.
    info : dict
        Diagnostic info: 'p_x', 'p_y', 'w_raw', weight statistics.
    """
    from scipy.stats import gaussian_kde

    N, d = features.shape

    ##### Subsample indices for KDE fitting #####
    if N > kde_subsample:
        rng = np.random.RandomState(0)
        fit_idx = rng.choice(N, size=kde_subsample, replace=False)
        print(f"  KDE fitting on {kde_subsample}/{N} subsampled points")
    else:
        fit_idx = np.arange(N)

    ##### Estimate p_x: input density #####
    if mode == 'full':
        # Product of independent 1D marginal KDEs (avoids curse of
        # dimensionality in high-d multivariate KDE).
        print(f"  Fitting p_x via {d} independent 1D marginal KDEs ...",
              end=" ", flush=True)
        log_p_x = np.zeros(N)
        for j in range(d):
            kde_j = gaussian_kde(features[fit_idx, j], bw_method=kde_bw)
            p_j = kde_j.evaluate(features[:, j])
            p_j = np.maximum(p_j, 1e-300)
            log_p_x += np.log(p_j)
        p_x = np.exp(log_p_x)
        p_x = np.maximum(p_x, 1e-300)
        print("done")
    else:
        # Uniform p_x → w ∝ 1/p_y
        print(f"  mode='output_only': assuming uniform input density")
        p_x = np.ones(N)

    ##### Estimate p_y: output density (univariate KDE) #####
    print(f"  Fitting p_y KDE (1D, {len(fit_idx)} points) ...", end=" ",
          flush=True)
    kde_y = gaussian_kde(targets[fit_idx], bw_method=kde_bw)
    p_y = kde_y.evaluate(targets)              # (N,) — 1D is fast
    p_y = np.maximum(p_y, 1e-300)
    print("done")

    ##### Likelihood ratio #####
    w_raw = p_x / p_y

    ##### Clip at quantile to remove extreme outliers, then normalise #####
    # Using a percentile-based clip avoids the re-normalisation instability
    # that occurs when a hard clip + renorm cycle amplifies residual weights.
    clip_pct = np.percentile(w_raw, 99)
    if clip_pct > 0:
        n_clipped = int(np.sum(w_raw > clip_pct * clip_max))
        weights = np.clip(w_raw, 0, clip_pct * clip_max)
    else:
        n_clipped = 0
        weights = w_raw.copy()

    ##### Normalise so mean = 1 #####
    w_mean = np.mean(weights)
    if w_mean > 0:
        weights = weights / w_mean
    else:
        weights = np.ones(N, dtype=np.float64)

    info = {
        'p_x': p_x,
        'p_y': p_y,
        'w_raw': w_raw,
        'w_mean': float(np.mean(weights)),
        'w_std': float(np.std(weights)),
        'w_min': float(np.min(weights)),
        'w_max': float(np.max(weights)),
        'w_median': float(np.median(weights)),
        'n_clipped': n_clipped,
    }

    print(f"  OW weights: mean={info['w_mean']:.3f}, "
          f"std={info['w_std']:.3f}, "
          f"min={info['w_min']:.3f}, max={info['w_max']:.3f}, "
          f"median={info['w_median']:.3f}, clipped={n_clipped}")

    return weights.astype(np.float32), info


def compute_iw_weights(targets, clip_max=50.0, power=1):
    """Compute inverse-weighted sample weights:  w_i = 1 / y_true_i^power.

    Upweights low-aTTC (dangerous) samples, downweights high-aTTC (safe)
    samples.  `power` controls how aggressively the boundary region is
    emphasized:

        power=1 -> w = 1/tau                (existing iw_huber)
        power=2 -> w = 1/tau^2              (iw_huber2)
        power=3 -> w = 1/tau^3              (iw_huber3)
        power=4 -> w = 1/tau^4              (iw_huber4)

    Higher powers concentrate more loss mass near the aTTC = tau_min
    decision boundary, at the cost of larger weight variance and more
    clipping at the tail.

    Parameters
    ----------
    targets : ndarray (N,)
        Positive target values (e.g. aTTC in seconds).
    clip_max : float
        After normalising weights to mean=1, clip at this multiple of the
        99th percentile to limit extreme outlier weights.  Default 50.
    power : int or float
        Reciprocal exponent.  Default 1 (matches the historic iw_huber).

    Returns
    -------
    weights : ndarray (N,) float32
        Sample weights normalised to mean=1.
    info : dict
        Diagnostic statistics: w_mean, w_std, w_min, w_max, w_median, n_clipped.
    """
    eps = 1e-8
    raw = 1.0 / (np.maximum(targets, 0) + eps) ** float(power)

    # Percentile-based clipping then normalise to mean=1
    p99 = np.percentile(raw, 99)
    clip_val = p99 * clip_max
    n_clipped = int((raw > clip_val).sum())
    weights = np.clip(raw, 0, clip_val)
    weights = weights / (weights.mean() + eps)

    info = {
        "w_mean":   float(weights.mean()),
        "w_std":    float(weights.std()),
        "w_min":    float(weights.min()),
        "w_max":    float(weights.max()),
        "w_median": float(np.median(weights)),
        "n_clipped": n_clipped,
        "power":    float(power),
    }

    print(f"  IW weights (power={power}): mean={info['w_mean']:.3f}, "
          f"std={info['w_std']:.3f}, "
          f"min={info['w_min']:.3f}, max={info['w_max']:.3f}, "
          f"median={info['w_median']:.3f}, clipped={n_clipped}")

    return weights.astype(np.float32), info


# ============================================================================
#                          LOSS  FUNCTIONS
# ============================================================================

def parse_loss_name(name):
    """Parse a loss name, separating any weighting prefix from the base loss.

    Supported prefixes:
        'ow_'  — output-weighted (Blanchard & Sapsis 2021)
        'iw_'  — inverse-weighted  (w = 1/y_true^power)

    A trailing integer on an 'iw_'-prefixed name sets the reciprocal
    exponent:

        'iw_huber'   -> base='huber', weighting='iw', power=1   (1/tau)
        'iw_huber2'  -> base='huber', weighting='iw', power=2   (1/tau^2)
        'iw_huber3'  -> base='huber', weighting='iw', power=3   (1/tau^3)
        'iw_huber4'  -> base='huber', weighting='iw', power=4   (1/tau^4)
        'iw_mse'     -> base='mse',   weighting='iw', power=1

    Returns
    -------
    base_loss : str
        The base loss name (e.g. 'mse', 'mae', 'huber', 'mape', 'log_mse').
    weighting : str or None
        'ow', 'iw', or None.
    power : int
        Reciprocal exponent for IW weighting.  Always 1 for non-IW losses
        or for IW names without a trailing integer.
    """
    import re
    name = name.lower().strip()
    if name.startswith("ow_"):
        return name[3:], "ow", 1
    if name.startswith("iw_"):
        rest = name[3:]
        # Optional trailing integer (e.g. 'huber2', 'mse4') sets the power.
        m = re.match(r"^([a-z_]+?)(\d+)?$", rest)
        if m and m.group(2):
            return m.group(1), "iw", int(m.group(2))
        return rest, "iw", 1
    return name, None, 1


def get_loss(name, huber_delta=1.0):
    """Return a Keras-compatible loss by name string.

    Supports base losses: mse, mae, huber, mape, log_mse.
    Prefix with 'ow_' for output-weighted variant (e.g. 'ow_mse').
    The OW weighting is applied via sample_weight in model.fit(),
    so this function returns the same base loss for both variants.

    Parameters
    ----------
    name : str
        Loss function name.
    huber_delta : float
        Threshold for Huber loss where it transitions from quadratic
        to linear.  Default is 1.0 (Keras default).  Higher values
        keep the quadratic regime for larger errors.
    """
    import tensorflow as tf

    # Strip OW/IW prefix and any IW power suffix — weighting handled externally
    base_name, _, _ = parse_loss_name(name)

    if base_name == "huber":
        return tf.keras.losses.Huber(delta=huber_delta)

    builtin = {
        "mse":   "mean_squared_error",
        "mae":   "mean_absolute_error",
        "mape":  "mean_absolute_percentage_error",
    }
    if base_name in builtin:
        return builtin[base_name]

    if base_name in ("log_mse", "logmse"):
        def log_mse(y_true, y_pred):
            eps = 1e-6
            return tf.reduce_mean(
                tf.square(tf.math.log(y_true + eps) - tf.math.log(y_pred + eps))
            )
        return log_mse

    all_losses = list(builtin.keys()) + ['log_mse']
    all_variants = (all_losses
                    + [f'ow_{l}' for l in all_losses]
                    + [f'iw_{l}' for l in all_losses])
    raise ValueError(
        f"Unknown loss: {name}. Choose from: {all_variants}"
    )
