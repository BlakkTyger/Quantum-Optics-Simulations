"""
Disk cache system for expensive computations.

Design:
    - Human-readable JSON metadata files describe each cached result
    - Numpy arrays stored in companion .npz files
    - Logical versioning: v1, v2, v3, etc. (set per-function)
    - Descriptive file names: {function_name}_v{version}_{param_summary}.json
    - Version mismatches trigger recomputation and new version files
"""

import os
import json
import inspect
from functools import wraps
from datetime import datetime
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache')


def _summarize_params(args, kwargs, func) -> str:
    """
    Create a short human-readable parameter summary for the cache filename.
    Uses the function's signature to map positional args to param names.
    """
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
    except Exception:
        param_names = []

    parts = []
    for i, val in enumerate(args):
        name = param_names[i] if i < len(param_names) else f"arg{i}"
        if isinstance(val, np.ndarray):
            parts.append(f"{name}=array({val.shape})_{np.sum(val):.0f}")
        elif isinstance(val, (int, float)):
            # Format numbers compactly
            if isinstance(val, float):
                parts.append(f"{name}={val:.4g}")
            else:
                parts.append(f"{name}={val}")
        elif isinstance(val, str):
            parts.append(f"{name}={val[:20]}")
        # Skip complex args to keep filename short

    for key, val in sorted(kwargs.items()):
        if isinstance(val, np.ndarray):
            parts.append(f"{key}=array({val.shape})_{np.sum(val):.0f}")
        elif isinstance(val, (int, float)):
            if isinstance(val, float):
                parts.append(f"{key}={val:.4g}")
            else:
                parts.append(f"{key}={val}")
        elif isinstance(val, tuple) and len(val) == 2:
            parts.append(f"{key}=({val[0]:.3g},{val[1]:.3g})")
        elif isinstance(val, bool):
            if val:
                parts.append(f"{key}=T")

    summary = "_".join(parts[:15])  # Include all params to avoid cache collisions
    # Clean invalid filename characters
    summary = summary.replace('/', '').replace('\\', '').replace(' ', '')
    summary = summary.replace('(', '').replace(')', '').replace(',', '-')
    # Limit total length
    if len(summary) > 120:
        summary = summary[:120]
    return summary


def _numpy_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        if obj.size <= 100:
            return {"__type__": "ndarray_small", "data": obj.tolist(),
                    "shape": list(obj.shape), "dtype": str(obj.dtype)}
        else:
            return {"__type__": "ndarray_file", "shape": list(obj.shape),
                    "dtype": str(obj.dtype), "size": int(obj.size)}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_json_serializable(item) for item in obj]
    return obj


def _save_result(cache_base, result, metadata):
    """
    Save computation result: JSON metadata + .npz for large arrays.
    """
    os.makedirs(os.path.dirname(cache_base), exist_ok=True)

    json_path = cache_base + ".json"
    npz_path = cache_base + ".npz"

    # Collect large numpy arrays separately
    arrays_to_save = {}
    json_result = {}

    if isinstance(result, dict):
        for key, val in result.items():
            if isinstance(val, np.ndarray) and val.size > 100:
                arrays_to_save[key] = val
                json_result[key] = {"__type__": "ndarray_file", "key": key,
                                    "shape": list(val.shape), "dtype": str(val.dtype)}
            else:
                json_result[key] = _numpy_to_json_serializable(val)
    elif isinstance(result, tuple):
        for i, val in enumerate(result):
            key = f"tuple_{i}"
            if isinstance(val, np.ndarray) and val.size > 100:
                arrays_to_save[key] = val
                json_result[key] = {"__type__": "ndarray_file", "key": key,
                                    "shape": list(val.shape), "dtype": str(val.dtype)}
            else:
                json_result[key] = _numpy_to_json_serializable(val)
        json_result["__result_type__"] = "tuple"
        json_result["__tuple_length__"] = len(result)
    elif isinstance(result, np.ndarray):
        if result.size > 100:
            arrays_to_save["result"] = result
            json_result["result"] = {"__type__": "ndarray_file", "key": "result",
                                     "shape": list(result.shape), "dtype": str(result.dtype)}
        else:
            json_result["result"] = _numpy_to_json_serializable(result)
        json_result["__result_type__"] = "ndarray"
    else:
        json_result["result"] = _numpy_to_json_serializable(result)
        json_result["__result_type__"] = "scalar"

    # Save metadata + result structure to JSON
    output = {
        "metadata": metadata,
        "result": json_result
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save large arrays to .npz
    if arrays_to_save:
        np.savez_compressed(npz_path, **arrays_to_save)


def _load_result(cache_base):
    """
    Load cached result from JSON + .npz files.
    """
    json_path = cache_base + ".json"
    npz_path = cache_base + ".npz"

    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data["metadata"]
    json_result = data["result"]

    # Load large arrays if present
    arrays = {}
    if os.path.exists(npz_path):
        npz_data = np.load(npz_path)
        arrays = dict(npz_data)
        npz_data.close()

    result_type = json_result.get("__result_type__", "dict")

    if result_type == "tuple":
        n = json_result["__tuple_length__"]
        items = []
        for i in range(n):
            key = f"tuple_{i}"
            val = json_result[key]
            if isinstance(val, dict) and val.get("__type__") == "ndarray_file":
                items.append(arrays[val["key"]])
            elif isinstance(val, dict) and val.get("__type__") == "ndarray_small":
                items.append(np.array(val["data"], dtype=val["dtype"]).reshape(val["shape"]))
            else:
                items.append(val)
        return metadata, tuple(items)
    elif result_type == "ndarray":
        val = json_result["result"]
        if isinstance(val, dict) and val.get("__type__") == "ndarray_file":
            return metadata, arrays[val["key"]]
        elif isinstance(val, dict) and val.get("__type__") == "ndarray_small":
            return metadata, np.array(val["data"], dtype=val["dtype"]).reshape(val["shape"])
        return metadata, np.array(val)
    elif result_type == "scalar":
        return metadata, json_result["result"]
    else:
        # dict result
        result = {}
        for key, val in json_result.items():
            if key.startswith("__"):
                continue
            if isinstance(val, dict) and val.get("__type__") == "ndarray_file":
                result[key] = arrays[val["key"]]
            elif isinstance(val, dict) and val.get("__type__") == "ndarray_small":
                result[key] = np.array(val["data"], dtype=val["dtype"]).reshape(val["shape"])
            else:
                result[key] = val
        return metadata, result


def disk_cache(version="v1"):
    """
    Decorator to cache function results to disk in human-readable format.

    Cache files are named descriptively:
        {function_name}_v{version}_{param_summary}.json  (metadata + small arrays)
        {function_name}_v{version}_{param_summary}.npz   (large arrays)

    Parameters
    ----------
    version : str
        Logical version string (e.g., "v1", "v2"). Bump this when the
        function's logic changes to invalidate old caches.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR, exist_ok=True)

            func_version = str(version)
            param_summary = _summarize_params(args, kwargs, func)

            # Build descriptive cache filename
            cache_name = f"{func.__name__}_{func_version}_{param_summary}"
            cache_base = os.path.join(CACHE_DIR, cache_name)
            json_path = cache_base + ".json"

            # Try loading from cache
            if os.path.exists(json_path):
                try:
                    metadata, result = _load_result(cache_base)
                    if metadata.get("version") == func_version:
                        print(f"[CACHE] Loaded: {cache_name}")
                        return result
                    else:
                        print(f"[CACHE] Version mismatch for {func.__name__} "
                              f"(cached={metadata.get('version')}, current={func_version}), "
                              f"recomputing...")
                except Exception as e:
                    print(f"[CACHE] Failed to load {cache_name}: {e}")

            # Compute result
            result = func(*args, **kwargs)

            # Build metadata
            metadata = {
                "function": func.__name__,
                "module": func.__module__,
                "version": func_version,
                "computed_at": datetime.now().isoformat(),
                "parameters": {}
            }

            # Store human-readable parameters
            try:
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                for i, val in enumerate(args):
                    name = param_names[i] if i < len(param_names) else f"arg{i}"
                    if isinstance(val, np.ndarray):
                        metadata["parameters"][name] = f"ndarray shape={val.shape} dtype={val.dtype}"
                    else:
                        metadata["parameters"][name] = _numpy_to_json_serializable(val)
                for key, val in kwargs.items():
                    if isinstance(val, np.ndarray):
                        metadata["parameters"][key] = f"ndarray shape={val.shape} dtype={val.dtype}"
                    else:
                        metadata["parameters"][key] = _numpy_to_json_serializable(val)
            except Exception:
                pass

            # Save to cache
            try:
                _save_result(cache_base, result, metadata)
                print(f"[CACHE] Saved: {cache_name}")
            except Exception as e:
                print(f"[CACHE] Failed to save {cache_name}: {e}")

            return result
        return wrapper
    return decorator
