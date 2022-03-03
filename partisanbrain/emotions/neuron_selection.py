import numpy as np


N_NEURONS = 100
MODEL_SHAPE = (49, 1600)
FILENAME = "output/output.npz"


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


def get_pos_neg(neuron_index, pos_samples, neg_samples):
    pos_sample = pos_samples[neuron_index]
    neg_sample = neg_samples[neuron_index]

    if pos_sample.mean() < neg_sample.mean():
        pos_activation = pos_sample.min()
        neg_activation = neg_sample.max()
    else:
        pos_activation = pos_sample.max()
        neg_activation = neg_sample.min()

    return pos_activation, neg_activation


def get_neurons_per_layer(neuron_indices, pos_samples, neg_samples):
    neurons_per_layer = {}
    for neuron_index in neuron_indices:
        layer, neuron = np.unravel_index(neuron_index, MODEL_SHAPE)
        layer, neuron = int(layer), int(neuron)

        pos_activation, neg_activation = get_pos_neg(
            neuron_index, pos_samples, neg_samples
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
    filename=FILENAME, n_neurons=N_NEURONS, method="random", sample_info=None
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

    neurons_per_layer = get_neurons_per_layer(neuron_indices, pos_samples, neg_samples)

    return neurons_per_layer
