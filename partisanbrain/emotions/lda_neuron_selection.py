import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import linalg as la
import torch


MODEL_SHAPE = (49, 1600)
FILENAME = "output/output.npz"


class LdaNeuronSelector:
    def __init__(self, filename=None, X=None, y=None, percentile=0.8):
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
        self.percentile = percentile

    def get_lda_neuron_dict(self, layer):
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.X[:, layer], self.y)

        projection = lda.scalings_
        projection = projection / la.norm(projection)

        dist = self.X[:, layer] @ projection
        pos_dist = dist[self.y == 1]
        neg_dist = dist[self.y == 0]

        if pos_dist.mean() > neg_dist.mean():
            pos_val = np.quantile(pos_dist, self.percentile)
            neg_val = np.quantile(neg_dist, 1 - self.percentile)
        else:
            pos_val = np.quantile(pos_dist, 1 - self.percentile)
            neg_val = np.quantile(neg_dist, self.percentile)

        return {
            "projection": torch.tensor(projection, dtype=torch.float),
            "positive": pos_val,
            "negative": neg_val,
        }

    def get_lda_neurons_per_layer(self, layers=25):
        if not isinstance(layers, list):
            layers = [layers]

        return {layer: self.get_lda_neuron_dict(layer) for layer in layers}
