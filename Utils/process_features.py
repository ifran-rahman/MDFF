import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA


def _chunks(data, batch_size, n_components):
    """Yield consecutive row chunks; chunks smaller than `n_components` are skipped
    (IncrementalPCA cannot fit them), as in the original pipeline."""
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        if len(chunk) >= n_components:
            yield i, chunk


def _fit_ipca(train_data, batch_size, n_components, extract_train_fn, ipca_path, checkpoint_every):
    """Pass 1: extract train features chunk by chunk and fit IncrementalPCA.

    The raw fused features (~200k dims per image) are never written to disk.
    A rolling checkpoint lets an interrupted run resume.
    """
    final_path = os.path.join(ipca_path, "ipca_conv.pkl")
    if os.path.exists(final_path):
        return joblib.load(final_path)

    ckpt_path = os.path.join(ipca_path, "ipca_checkpoint.pkl")
    ipca, start = IncrementalPCA(n_components=n_components), 0
    if os.path.exists(ckpt_path):
        ipca, start = joblib.load(ckpt_path)
        print(f"Resuming PCA fit from row {start}")

    chunks = [(i, c) for i, c in _chunks(train_data, batch_size, n_components) if i >= start]
    for k, (i, chunk) in enumerate(tqdm(chunks, desc="Pass 1/3: train features -> fit PCA"), 1):
        features = extract_train_fn(chunk)
        ipca.partial_fit(features[:, :-1])
        if k % checkpoint_every == 0:
            joblib.dump((ipca, i + batch_size), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    joblib.dump(ipca, final_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return ipca


def _reduce(data, batch_size, n_components, extract_fn, ipca, desc):
    """Extract features chunk by chunk and project them with the fitted PCA."""
    reduced = []
    for _, chunk in tqdm(list(_chunks(data, batch_size, n_components)), desc=desc):
        features = extract_fn(chunk)
        reduced.append(np.hstack((ipca.transform(features[:, :-1]), features[:, -1:])))
    return np.vstack(reduced)


def process_features(
    train_data,
    test_data,
    feature_dir,
    batch_size,
    n_components,
    model_save_name,
    extract_train_fn,
    extract_val_fn,
    checkpoint_every=50,
):
    """Feature extraction -> IncrementalPCA fit on train -> PCA-reduced train/val sets.

    Saves `<feature_dir>/train/<name>_train.npy` and `<feature_dir>/val/<name>_val.npy`
    (PCA components plus the target in the last column) and returns both arrays.
    If both files already exist they are loaded instead of recomputed.
    """
    ipca_path = os.path.join(feature_dir, "ipca")
    train_out = os.path.join(feature_dir, "train", f"{model_save_name}_train.npy")
    val_out = os.path.join(feature_dir, "val", f"{model_save_name}_val.npy")
    for d in (ipca_path, os.path.dirname(train_out), os.path.dirname(val_out)):
        os.makedirs(d, exist_ok=True)

    if os.path.exists(train_out) and os.path.exists(val_out):
        print("Loading cached PCA-reduced features")
        return np.load(train_out), np.load(val_out)

    ipca = _fit_ipca(train_data, batch_size, n_components, extract_train_fn, ipca_path, checkpoint_every)

    if not os.path.exists(train_out):
        np.save(train_out, _reduce(train_data, batch_size, n_components, extract_train_fn, ipca,
                                   "Pass 2/3: train features -> PCA"))
    np.save(val_out, _reduce(test_data, batch_size, n_components, extract_val_fn, ipca,
                             "Pass 3/3: val features -> PCA"))

    return np.load(train_out), np.load(val_out)
