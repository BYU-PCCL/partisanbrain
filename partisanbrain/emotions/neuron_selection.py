from multiprocessing.sharedctypes import Value
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.decomposition import PCA
import torch


MODEL_SHAPE = (49, 1600)
FILENAME = "output/output.npz"


class NeuronSelector:
    def __init__(self, input_filename=FILENAME, X=None, y=None, device=None):
        if input_filename is not None:
            self.set_samples(filename=input_filename)
        else:
            self.set_samples(X=X, y=y)
        self.device = device
        self.method = None

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
        self, n_neurons=100, percentile=0.8, method="correlation"
    ):
        if method != self.method or not hasattr(self, "neuron_indices"):
            self.rank_neuron_indices(method=method)
            self.method = method

        neurons_per_layer = {}
        for neuron_index in self.neuron_indices[:n_neurons]:
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
                transform = torch.tensor(self.transforms[layer], dtype=torch.float)

                # TODO maybe we should restructure this to only save one transform per layer
                if self.device:
                    transform = transform.to(self.device)
                neuron_dict["transform"] = transform

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

    def do_corr(self, X, layer):
        return [np.corrcoef(self.y, X[:, layer, i])[0, 1] for i in range(X.shape[2])]

    def rank_with_correlation(self, X=None):
        X = X if X is not None else self.X
        corrs = np.array([self.do_corr(X, layer) for layer in range(MODEL_SHAPE[0])])
        return np.abs(corrs).ravel().argsort()[::-1]

    def rank_with_log_reg(self, X=None):
        X = X if X is not None else self.X

        accs = []
        Xhat = X.reshape(-1, np.prod(X.shape[1:]))
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
        Xhat, self.transforms = self.get_transforms()
        return self.rank_with_correlation(Xhat)

    def rank_with_pca_log_reg(self):
        Xhat, self.transforms = self.get_transforms()
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


# def pca_transform(X, dim=1):
#     axes = np.arange(X.ndim)
#     new_axes = axes[dim : dim + 1] + axes[:dim] + axes[dim + 1 :]
#     # reverse_axes = list(range(dim)) + [0] + list(range(dim+1, X.ndim))

#     pcas = []
#     Xhat = []

#     for Xi in np.transpose(X, axes=new_axes):
#         pca = PCA(n_components=MODEL_SHAPE[-1])
#         Xhati = pca.fit_transform(Xi)
#         pcas.append(pca)
#         Xhat.append(Xhati)
#     Xhat = np.array(Xhati)

#     return Xhat, pcas


# def do_corr(layer, activations, targets):
#     return [
#         np.corrcoef(targets[:, 0], activations[:, layer, ind])[0, 1]
#         for ind in range(activations.shape[2])
#     ]


# def do_log_reg(activations, targets):
#     # do we want to make layer a parameter?, or do all layers at a time?
#     # obtains and returns accuracies for logistic regressions
#     accs = []
#     activations = activations.reshape(-1, np.prod(activations.shape[1:]))
#     for i in range(activations.shape[1]):
#         clf = LogisticRegression().fit(activations[:, i : i + 1], targets)
#         acc = clf.score(activations[:, i : i + 1], targets)
#         accs.append(acc)
#     return accs


# def get_samples(filename=FILENAME):
#     output = np.load(filename)

#     X = output["activations"]
#     y = output["targets"]

#     samples = X.reshape(y.shape[0], -1)
#     pos_mask = (y == 1).squeeze()
#     neg_mask = (y == 0).squeeze()
#     pos_samples = samples[pos_mask].T
#     neg_samples = samples[neg_mask].T

#     return X, y, samples, pos_samples, neg_samples


# def get_corrs(X, y):
#     return np.array([do_corr(layer, X, y) for layer in range(MODEL_SHAPE[0])])


# def get_force_values(neuron_index, pos_samples, neg_samples, percentile=0.75):
#     pos_sample = pos_samples[neuron_index]
#     neg_sample = neg_samples[neuron_index]

#     if pos_sample.mean() < neg_sample.mean():
#         pos_activation = np.quantile(pos_sample, 1 - percentile)
#         neg_activation = np.quantile(neg_sample, percentile)
#     else:
#         pos_activation = np.quantile(pos_sample, percentile)
#         neg_activation = np.quantile(neg_sample, 1 - percentile)

#     return pos_activation, neg_activation


# def get_neurons_per_layer(neuron_indices, pos_samples, neg_samples, percentile):
#     neurons_per_layer = {}
#     for neuron_index in neuron_indices:
#         layer, neuron = np.unravel_index(neuron_index, MODEL_SHAPE)
#         layer, neuron = int(layer), int(neuron)

#         pos_activation, neg_activation = get_force_values(
#             neuron_index, pos_samples, neg_samples, percentile=percentile
#         )

#         neuron_dict = {
#             "neuron": neuron,
#             "positive": float(pos_activation),
#             "negative": float(neg_activation),
#         }

#         if layer in neurons_per_layer:
#             neurons_per_layer[layer].append(neuron_dict)
#         else:
#             neurons_per_layer[layer] = [neuron_dict]
#     return neurons_per_layer


# def select_neurons_per_layer(
#     filename=FILENAME,
#     n_neurons=N_NEURONS,
#     method="correlation",
#     sample_info=None,
#     percentile=0.75,
# ):
#     if sample_info is None:
#         X, y, samples, pos_samples, neg_samples = get_samples(filename)
#     else:
#         X, y, samples, pos_samples, neg_samples = sample_info

#     if method == "correlation":
#         corrs = get_corrs(X, y)
#         corray = np.array(corrs).ravel()
#         neuron_indices = np.abs(corray).argsort()[::-1][:n_neurons]
#     elif method == "logistic_regression":
#         accs = do_log_reg(X, y)
#         neuron_indices = np.argsort(accs)[::-1][:n_neurons]
#     elif method == "pca_correlation":
#         pass
#     elif method == "pca_log_reg":
#         pass
#     elif method == "random":
#         neuron_indices = np.random.choice(
#             a=np.prod(MODEL_SHAPE), size=n_neurons, replace=False
#         )
#     else:
#         raise ValueError("method must equal either 'random' or 'correlation'")

#     neurons_per_layer = get_neurons_per_layer(
#         neuron_indices, pos_samples, neg_samples, percentile=percentile
#     )

#     return neurons_per_layer
