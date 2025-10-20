import pickle
import os
import pathlib


def atomic_pickle_dump(obj, target_path: str):
    target = pathlib.Path(target_path)
    tmp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}-{int(__import__('time').time()*1000)}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def safe_pickle_load(path: str):
    try:
        if (not os.path.isfile(path)) or (os.path.getsize(path) == 0):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
