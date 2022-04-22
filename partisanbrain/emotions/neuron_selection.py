from multiprocessing.sharedctypes import Value
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.decomposition import PCA
import torch


MODEL_SHAPE = (49, 1600)
FILENAME = "output/sentiment_activations.npz"


class NeuronSelector:
    def __init__(self, input_filename=FILENAME, X=None, y=None, device=None):
        if input_filename is not None:
            self.set_samples(filename=input_filename)
        else:
            self.set_samples(X=X, y=y)
        self.device = device
        self.method = None
        self.transforms = None

    def set_samples(self, filename=None, X=None, y=None):
        if filename is not None:
            output = np.load(filename)

        self.X = output["activations"] if X is None else X
        self.y = output["targets"] if y is None else y
        self.y = self.y.squeeze()

        samples = self.X.reshape(self.y.shape[0], -1)
        pos_mask = (self.y == 1).squeeze()
        neg_mask = (self.y == 0).squeeze()
        self.pos_samples = samples[pos_mask].T
        self.neg_samples = samples[neg_mask].T

    def get_neurons_per_layer(
        self, n_neurons=100, percentile=0.8, method="correlation", layer=None
    ):
        if method != self.method or not hasattr(self, "neuron_indices"):
            self.rank_neuron_indices(method=method)
            self.method = method

        if layer is None:
            filtered_neuron_indices = self.neuron_indices[:n_neurons]
        else:
            start_index = layer * MODEL_SHAPE[-1]
            end_index = (layer + 1) * MODEL_SHAPE[-1]
            mask = self.neuron_indices >= start_index & self.neuron_indices < end_index
            filtered_neuron_indices = self.neuron_indices[mask][:n_neurons]

        neurons_per_layer = {}
        for neuron_index in filtered_neuron_indices:
            layer, neuron = np.unravel_index(neuron_index, MODEL_SHAPE)
            layer, neuron = int(layer), int(neuron)

            pos_activation, neg_activation = self.get_force_values(
                neuron_index=neuron_index, percentile=percentile
            )

            neuron_dict = {
                "neuron": neuron,
                "positive": float(pos_activation),
                "negative": float(neg_activation),
            }

            if self.method.startswith("pca"):
                if layer in neurons_per_layer:
                    neurons_per_layer[layer]["dicts"].append(neuron_dict)
                else:
                    transform = torch.tensor(self.transforms[layer], dtype=torch.float)
                    if self.device:
                        transform = transform.to(self.device)

                    neurons_per_layer[layer] = {
                        "transform": transform,
                        "dicts": [neuron_dict],
                    }
            else:
                if layer in neurons_per_layer:
                    neurons_per_layer[layer].append(neuron_dict)
                else:
                    neurons_per_layer[layer] = [neuron_dict]
        return neurons_per_layer

    def rank_neuron_indices(self, method):
        if method == "correlation":
            self.neuron_indices = self.rank_with_correlation()
        elif method == "logistic_regression":
            self.neuron_indices = self.rank_with_log_reg()
        elif method == "pca_correlation":
            self.neuron_indices = self.rank_with_pca_correlation()
        elif method == "pca_log_reg":
            self.neuron_indices = self.rank_with_pca_log_reg()
        else:
            raise ValueError(
                "method must equal either 'correlation', 'logistic_regression', 'pca_correlation' or 'pca_log_reg'"
            )

    def get_corr(self, X, layer):
        return [np.corrcoef(self.y, X[:, layer, i])[0, 1] for i in range(X.shape[2])]

    def rank_with_correlation(self, X=None):
        X = X if X is not None else self.X
        corrs = np.array([self.get_corr(X, layer) for layer in range(MODEL_SHAPE[0])])
        return np.abs(corrs).ravel().argsort()[::-1]

    def rank_with_log_reg(self, X=None):
        X = X if X is not None else self.X

        accs = []
        Xhat = X.reshape(X.shape[0], -1)
        model = LogisticRegression()
        for i in range(Xhat.shape[1]):
            model.fit(Xhat[:, i : i + 1], self.y)
            acc = model.score(Xhat[:, i : i + 1], self.y)
            accs.append(acc)
        return np.argsort(accs)[::-1]

    def get_transforms(self):
        # axes = np.arange(self.X.ndim)
        # new_axes = axes[dim : dim + 1] + axes[:dim] + axes[dim + 1 :]
        # reverse_axes = list(range(dim)) + [0] + list(range(dim+1, X.ndim))
        axes = (1, 0, 2)

        transforms = []
        Xhat = []

        # Calculate and save the transform for each layer
        for Xi in np.transpose(self.X, axes=axes):
            pca = PCA(n_components=MODEL_SHAPE[-1])
            pca.fit(Xi)
            # We don't shift by the means because we don't need that functionality
            Xhati = np.dot(Xi, pca.components_.T)
            transforms.append(pca.components_.T)
            Xhat.append(Xhati)

        Xhat = np.transpose(np.array(Xhat), axes=axes)

        self.set_samples(X=Xhat, y=self.y)

        return Xhat, transforms

    def rank_with_pca_correlation(self):
        if self.transforms is None:
            Xhat, self.transforms = self.get_transforms()
        else:
            Xhat = self.X

        return self.rank_with_correlation(Xhat)

    def rank_with_pca_log_reg(self):
        if self.transforms is None:
            Xhat, self.transforms = self.get_transforms()
        else:
            Xhat = self.X

        return self.rank_with_log_reg(Xhat)

    def get_force_values(self, neuron_index, percentile):
        pos_sample = self.pos_samples[neuron_index]
        neg_sample = self.neg_samples[neuron_index]

        if pos_sample.mean() < neg_sample.mean():
            pos_activation = np.quantile(pos_sample, 1 - percentile)
            neg_activation = np.quantile(neg_sample, percentile)
        else:
            pos_activation = np.quantile(pos_sample, percentile)
            neg_activation = np.quantile(neg_sample, 1 - percentile)

        return pos_activation, neg_activation
