from multiprocessing.sharedctypes import Value
from sklearn.linear_model import LogisticRegression
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
    return [
        np.corrcoef(targets[:, 0], activations[:, layer, ind])[0, 1]
        for ind in range(activations.shape[2])
    ]


def do_log_reg(activations, targets):
    # do we want to make layer a parameter?, or do all layers at a time?
    # obtains and returns accuracies for logistic regressions
    accs = []
    activations = activations.reshape(-1, np.prod(activations.shape[1:]))
    for i in range(activations.shape[1]):
        clf = LogisticRegression().fit(activations[:, i : i + 1], targets)
        acc = clf.score(activations[:, i : i + 1], targets)
        accs.append(acc)
    return accs


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
    return np.array([do_corr(layer, X, y) for layer in range(MODEL_SHAPE[0])])


def get_force_values(neuron_index, pos_samples, neg_samples, percentile=0.75):
    pos_sample = pos_samples[neuron_index]
    neg_sample = neg_samples[neuron_index]

    if pos_sample.mean() < neg_sample.mean():
        pos_activation = np.quantile(pos_sample, 1 - percentile)
        neg_activation = np.quantile(neg_sample, percentile)
    else:
        pos_activation = np.quantile(pos_sample, percentile)
        neg_activation = np.quantile(neg_sample, 1 - percentile)

    return pos_activation, neg_activation


def get_neurons_per_layer(neuron_indices, pos_samples, neg_samples, percentile):
    neurons_per_layer = {}
    for neuron_index in neuron_indices:
        layer, neuron = np.unravel_index(neuron_index, MODEL_SHAPE)
        layer, neuron = int(layer), int(neuron)

        pos_activation, neg_activation = get_force_values(
            neuron_index, pos_samples, neg_samples, percentile=percentile
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
    percentile=0.75,
):
    if sample_info is None:
        X, y, samples, pos_samples, neg_samples = get_samples(filename)
    else:
        X, y, samples, pos_samples, neg_samples = sample_info

    if method == "correlation":
        corrs = get_corrs(X, y)
        corray = np.array(corrs).ravel()
        neuron_indices = np.abs(corray).argsort()[::-1][:n_neurons]
    elif method == "logistic_regression":
        # I ADDED THIS STUFF TOO
        accs = do_log_reg(X, y)
        neuron_indices = np.argsort(accs)[::-1][:n_neurons]
    elif method == "random":
        neuron_indices = np.random.choice(
            a=np.prod(MODEL_SHAPE), size=n_neurons, replace=False
        )
    else:
        raise ValueError("method must equal either 'random' or 'correlation'")

    neurons_per_layer = get_neurons_per_layer(
        neuron_indices, pos_samples, neg_samples, percentile=percentile
    )

    return neurons_per_layer
