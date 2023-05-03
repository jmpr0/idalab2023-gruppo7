import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer

def duplicating(x):
    return jittering(x, u=0, s=0, features='all')

def jittering(x, u=0, s=.03, features='all', random_state=None):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    mask = np.zeros_like(x)
    jittering_matrix = execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=x.shape), random_state)
    mask[..., features] = jittering_matrix[..., features]
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 0
    x_tr = np.abs(x + mask).astype(x.dtype)
    x_tr[np.where(x_tr > 1)] = 1.
    return x_tr

def traslating(x, u=0, s=.03, features='all', random_state=None):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    while True:
        mask = np.zeros_like(x)
        traslation_matrix = np.ones_like(x) * execute_revert_random_state(
            np.random.normal, dict(loc=u, scale=s, size=None), random_state)
        mask[..., features] = traslation_matrix[..., features]
        # Forcing the first IAT to the original value
        mask[:, 0, 1] = 0
        if ((x + mask) >= 0).all():
            break
    x_tr = (x + mask).astype(x.dtype)
    x_tr[np.where(x_tr > 1)] = 1.
    return x_tr

def scaling(x, u=1, s=.2, features='all', random_state=None):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    mask = np.ones_like(x)
    scaling_matrix = np.ones_like(x) * execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=None), random_state)
    mask[..., features] = scaling_matrix[..., features]
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 1.
    x_tr = (x * mask).astype(x.dtype)
    x_tr[np.where(x_tr > 1)] = 1.
    return x_tr 

def warping(x, u=1, s=.2, n_knot=4, features='all', random_state=None):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    knots = execute_revert_random_state(
        np.random.normal, dict(loc=u, scale=s, size=n_knot), random_state)
    n_points = x.shape[1]
    mask = np.ones_like(x)
    warping_matrix = np.interp(
        np.arange(n_points), np.linspace(0, n_points - 1, n_knot), knots
        ).repeat(x.shape[-1], axis=-1).reshape(x.shape)
    mask[..., features] = warping_matrix[..., features]
    # Forcing the first IAT to the original value
    mask[:, 0, 1] = 1.
    x_tr = (x * mask).astype(x.dtype)
    x_tr[np.where(x_tr > 1)] = 1.
    return x_tr

def slicing(x, wr=.9, pr=None, fill_value=0, features='all'):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        features = list(range(x.shape[-1]))
    if not isinstance(fill_value, (np.ndarray, list, set)):
        fill_value = [fill_value] * x.shape[-1]
    n_points = x.shape[1]
    w = max(int(np.floor(n_points * wr)), 1)
    pos = np.random.randint(n_points) if pr is None else int(pr * n_points)
    sliced_x = deepcopy(x)
    if w + pos <= n_points:
        sliced_x[..., :pos, features] = np.nan
        sliced_x[..., pos + w:, features] = np.nan
    else:
        sliced_x[..., pos + w - n_points:pos, features] = np.nan
    fill_matrix = np.repeat(fill_value, n_points).reshape(x.shape[1:][::-1]).T.reshape(x.shape)
    sliced_x[np.isnan(sliced_x)] = fill_matrix[np.isnan(sliced_x)]
    return sliced_x

def features_hiding(x, features='all', fill_value=0):
    check_features(features)
    if isinstance(features, str) and features == 'none':
        return deepcopy(x)
    if isinstance(features, str) and features == 'all':
        return (np.ones_like(x) * fill_value).astype(x.dtype)
    mask = np.ones_like(x)
    mask[..., features] = 0
    return (x * mask + fill_value * (1 - mask)).astype(x.dtype)

def check_features(features):
    if isinstance(features, str):
        assert features in ['none', 'all']
    else:
        assert isinstance(features, (np.ndarray, list, set))

def execute_revert_random_state(fn, fn_kwargs=None, new_random_state=None):
    """
    Execute fn(**fn_kwargs) without impacting the external random_state behavior.
    """
    old_random_state = np.random.get_state()
    np.random.seed(new_random_state)
    ret = fn(**fn_kwargs)
    np.random.set_state(old_random_state)
    return ret
