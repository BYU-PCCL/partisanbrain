import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import linalg as la
from sklearn.linear_model import LogisticRegression
import torch


MODEL_SHAPE = (49, 1600)
FILENAME = "output/sentiment_activations.npz"


class LdaNeuronSelector:
    def __init__(
        self,
        filename=None,
        device=None,
        X=None,
        y=None,
    ):
        """
        Parameters:
            X (ndarray): Should be of shape (batch_size x layers x neurons)
            y (ndarray): Should be of shape (batch_size)
            percentile (float): To what percentile do we want to push the virtual sentiment neuron
        """
        if filename is None:
            if X is None or y is None:
                raise ValueError(
                    "You must specity either 'filename' or both 'X' and 'y'"
                )

            self.X = X
            self.y = y.squeeze()
        else:
            output = np.load(filename)

            self.X = output["activations"]
            self.y = output["targets"].squeeze()

        self.device = device

    def get_lda_neuron_dict(self, layer, percentile):
        projection = self.projections[layer]
        dist = self.dists[layer]

        pos_dist = dist[self.y == 1]
        neg_dist = dist[self.y == 0]

        if pos_dist.mean() > neg_dist.mean():
            pos_val = np.quantile(pos_dist, percentile)
            neg_val = np.quantile(neg_dist, 1 - percentile)
        else:
            pos_val = np.quantile(pos_dist, 1 - percentile)
            neg_val = np.quantile(neg_dist, percentile)

        projection = torch.tensor(projection, dtype=torch.float)

        if self.device:
            projection = projection.to(self.device)

        return {
            "projection": projection,
            "positive": pos_val,
            "negative": neg_val,
        }

    def select_with_log_reg(self):
        accs = []
        model = LogisticRegression()
        for dist in self.dists:
            model.fit(dist, self.y)
            acc = model.score(dist, self.y)
            accs.append(acc)
        return np.argsort(accs)[::-1]

    def select_with_correlation(self):
        corrs = [np.corrcoef(self.y, dist.squeeze())[0, 1] for dist in self.dists]
        return np.argsort(corrs)[::-1]

    def set_projections_and_dists(self):
        self.projections = []
        self.dists = []
        lda = LinearDiscriminantAnalysis()
        for Xi in np.transpose(self.X, (1, 0, 2)):
            lda.fit(Xi, self.y)

            projection = lda.scalings_
            projection = projection / la.norm(projection)

            dist = Xi @ projection

            self.dists.append(dist)
            self.projections.append(projection)

    def select_layers(self, method, n_layers):
        if not hasattr(self, "projections") or not hasattr(self, "dists"):
            self.set_projections_and_dists()

        if method == "correlation":
            layer_rankings = self.select_with_correlation()
        elif method == "logistic_regression":
            layer_rankings = self.select_with_log_reg()

        self.layers = layer_rankings[:n_layers]

    def get_lda_neurons_per_layer(
        self, n_layers=10, percentile=0.8, method="correlation"
    ):
        self.select_layers(method=method, n_layers=n_layers)
        return {
            layer: self.get_lda_neuron_dict(layer=layer, percentile=percentile)
            for layer in self.layers
        }
