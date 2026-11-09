import numpy as np


def to_python(value):
    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_python(value.tolist())
    if isinstance(value, np.generic):
        return to_python(value.item())
    return value
