from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.decomposition import PCA


N_NEURONS = 100
MODEL_SHAPE = (49, 1600)
FILENAME = "output/output.npz"


def pca_transform(X, dim=1):
    axes = np.arange(X.ndim)
    new_axes = axes[dim : dim + 1] + axes[:dim] + axes[dim + 1 :]
    # reverse_axes = list(range(dim)) + [0] + list(range(dim+1, X.ndim))

    pcas = []
    Xhat = []

    for Xi in np.transpose(X, axes=new_axes):
        pca = PCA(n_components=MODEL_SHAPE[-1])
        Xhati = pca.fit_transform(Xi)
        pcas.append(pca)
        Xhat.append(Xhati)
    Xhat = np.array(Xhati)

    return Xhat, pcas


def do_corr(layer, activations, targets):
    corrs = []
    for ind in range(activations.shape[2]):
        result = np.corrcoef(targets[:, 0], activations[:, layer, ind])
        corrs.append(result[0, 1])
    return corrs


def get_samples(filename=FILENAME):
    output = np.load(filename)

    X = output["activations"]
    y = output["targets"]

    samples = X.reshape(y.shape[0], -1)
    pos_mask = (y == 1).squeeze()
    neg_mask = (y == 0).squeeze()
    pos_samples = samples[pos_mask].T
    neg_samples = samples[neg_mask].T

    return X, y, samples, pos_samples, neg_samples


def get_corrs(X, y):
    corrs = []
    for layer in range(MODEL_SHAPE[0]):
        corr = do_corr(layer, X, y)
        corrs.append(corr)

    return np.array(corrs)


def get_force_values(neuron_index, pos_samples, neg_samples, force_to="mean", percentile=0.25):
    pos_sample = pos_samples[neuron_index]
    neg_sample = neg_samples[neuron_index]

    if force_to == "mean":
        pos_activation = pos_sample.mean()
        neg_activation = neg_sample.mean()
    elif force_to == "extreme":
        if pos_sample.mean() < neg_sample.mean():
            pos_activation = pos_sample.min()
            neg_activation = neg_sample.max()
        else:
            pos_activation = pos_sample.max()
            neg_activation = neg_sample.min()
    elif force_to == "quantile":
        if pos_sample.mean() < neg_sample.mean():
            pos_activation = np.quantile(pos_sample, percentile)
            neg_activation = np.quantile(neg_sample, 1 - percentile)
        else:
            pos_activation = np.quantile(pos_sample, 1 - percentile)
            neg_activation = np.quantile(neg_sample, percentile)
    else:
        raise ValueError("force_to must be 'mean', 'extreme', or 'quantile'")

    return pos_activation, neg_activation


def get_neurons_per_layer(neuron_indices, pos_samples, neg_samples, force_to):
    neurons_per_layer = {}
    for neuron_index in neuron_indices:
        layer, neuron = np.unravel_index(neuron_index, MODEL_SHAPE)
        layer, neuron = int(layer), int(neuron)

        pos_activation, neg_activation = get_force_values(
            neuron_index, pos_samples, neg_samples, force_to=force_to
        )

        neuron_dict = {
            "neuron": neuron,
            "positive": float(pos_activation),
            "negative": float(neg_activation),
        }

        if layer in neurons_per_layer:
            neurons_per_layer[layer].append(neuron_dict)
        else:
            neurons_per_layer[layer] = [neuron_dict]
    return neurons_per_layer


def select_neurons_per_layer(
    filename=FILENAME,
    n_neurons=N_NEURONS,
    method="correlation",
    sample_info=None,
    force_to="mean",
):
    if sample_info is None:
        X, y, samples, pos_samples, neg_samples = get_samples(filename)
    else:
        X, y, samples, pos_samples, neg_samples = sample_info

    if method == "correlation":
        corrs = get_corrs(X, y)
        corray = np.array(corrs).ravel()
        neuron_indices = np.abs(corray).argsort()[::-1][:n_neurons]
    elif method == "random":
        neuron_indices = np.random.choice(
            a=np.prod(MODEL_SHAPE), size=n_neurons, replace=False
        )
    else:
        raise ValueError("method must equal either 'random' or 'correlation'")

    neurons_per_layer = get_neurons_per_layer(
        neuron_indices, pos_samples, neg_samples, force_to=force_to
    )

    return neurons_per_layer
