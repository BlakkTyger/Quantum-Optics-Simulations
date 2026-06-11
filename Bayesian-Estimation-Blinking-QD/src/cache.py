import os
import pickle
import hashlib
import inspect
from functools import wraps
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cache')

def get_function_version(func, custom_version=None):
    """
    Generate a version hash for a function based on its source code.
    If custom_version is provided, use that instead.
    """
    if custom_version is not None:
        return str(custom_version)
    try:
        source_code = inspect.getsource(func)
        return hashlib.md5(source_code.encode('utf-8')).hexdigest()
    except Exception:
        return "unknown_version"

def make_hashable(obj):
    """
    Convert unhashable objects (like numpy arrays) into hashable representations.
    """
    if isinstance(obj, np.ndarray):
        return ('__ndarray__', obj.shape, obj.dtype, obj.tobytes())
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)
    return obj

def disk_cache(version=None):
    """
    Decorator to cache function results to disk.
    The cache key is based on the function name, its source code version, and its arguments.
    Automatically invalidates if the function's source code changes.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR, exist_ok=True)
            
            func_version = get_function_version(func, version)
            
            # Serialize arguments to create a cache key
            hashable_args = make_hashable(args)
            hashable_kwargs = make_hashable(kwargs)
            
            cache_data = {
                'func_name': func.__name__,
                'module': func.__module__,
                'version': func_version,
                'args': hashable_args,
                'kwargs': hashable_kwargs
            }
            
            try:
                key_str = pickle.dumps(cache_data)
                key_hash = hashlib.md5(key_str).hexdigest()
                cache_file = os.path.join(CACHE_DIR, f"{func.__name__}_{key_hash}.pkl")
                
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_version, result = pickle.load(f)
                        if cached_version == func_version:
                            print(f"[CACHE] Loaded {func.__name__} from cache.")
                            return result
                        else:
                            print(f"[CACHE] Version mismatch for {func.__name__}, recomputing...")
                    except Exception as e:
                        print(f"[CACHE] Failed to load cache for {func.__name__}: {e}")
            except Exception as e:
                print(f"[CACHE] Warning: could not generate cache key for {func.__name__}: {e}")
                cache_file = None
            
            # If not cached or load failed, compute result
            result = func(*args, **kwargs)
            
            # Save to cache
            if cache_file:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump((func_version, result), f)
                except Exception as e:
                    print(f"[CACHE] Failed to save cache for {func.__name__}: {e}")
                    
            return result
        return wrapper
    return decorator
