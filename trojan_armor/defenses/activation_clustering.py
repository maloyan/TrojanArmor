from copy import deepcopy
from functools import partial

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader, Subset

class ActivationClusteringDefense:
    def __init__(self, model, layer, n_ica_components=5000, threshold: float = 0.4, device='cpu'):
        """
        Initializes the Activation Clustering defense method.

        :param model: model from which activations will be taken
        :param layer: The layer whose activations will be clustered
        :param n_ica_components: The number components for Independent Component Analysis (ICA)
        :param threshold: The threshold parameter to determine if the removed cluster is poisonous based on the classification ratio
        :param device: The device ('cpu' or 'cuda') to perform computations on
        """
        self.model = model.to(device)
        self.layer = layer
        self.kmeans_clusters = 2
        self.n_ica_components = n_ica_components
        assert 0 < threshold < 1, f'Threshold value expected between 0 and 1, got {threshold}'
        self.threshold = threshold
        self.device = device

    @staticmethod
    def _save_activation_hook(module, input, output, activations):
        """Hook that saves the output of the layer."""
        activations.append(output.detach())

    def _get_activations(self, dataloader: DataLoader):
        # Ensure the model is in evaluation mode
        self.model.eval()

        activations = []
        # Register a hook to save the activations of the specified layer
        hook = self.layer.register_forward_hook(partial(self._save_activation_hook, activations=activations))
        with torch.no_grad():
            for images, _ in dataloader:
                inputs = images.to(self.device)
                self.model(inputs)
        hook.remove()
        return activations

    def _reduce_dimensions(self, activations):
        # Convert list of activations to a tensor and reshape for clustering
        activations_tensor = torch.cat(activations, dim=0)

        # Reshape to (number of samples, feature size) for clustering
        activations_reshaped = activations_tensor.view(activations_tensor.size(0), -1).cpu().numpy()

        n_ica_components = min(self.n_ica_components, activations_reshaped.shape[1])
        transformer = FastICA(n_ica_components, random_state=0, whiten='unit-variance')
        activations_ica_reduced = transformer.fit_transform(activations_reshaped)
        return activations_ica_reduced

    def _get_kmeans_clusters(self, activations_ica_reduced):
        # Fit K-means clustering on the activations
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=0)
        labels = kmeans.fit_predict(activations_ica_reduced)
        return labels

    def _analyze_cluster_data(self, num_cluster: int, source_class: int, dataloader: DataLoader, new_model, clusters_labels, train_function, evaluate_function):
        images_in_cluster = Subset(dataloader.dataset, np.where(clusters_labels == num_cluster)[0])
        images_outside_cluster = Subset(dataloader.dataset, np.where(clusters_labels != num_cluster)[0])

        # Train model on examples from cluster=num_cluster
        images_in_cluster_loader = DataLoader(images_in_cluster, batch_size=dataloader.batch_size, shuffle=True)
        clean_model = deepcopy(new_model)
        train_function(clean_model, images_in_cluster_loader)

        # Predict class for examples from another cluster
        images_outside_cluster_loader = DataLoader(images_outside_cluster, batch_size=dataloader.batch_size, shuffle=False)
        y_true, y_pred = evaluate_function(clean_model, images_outside_cluster_loader)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        label_count = np.sum(y_true == y_pred)
        source_class_count = np.sum(y_pred == source_class)
        # Proportion of data classified as the source class over the data classified as their original label.
        ratio = label_count / source_class_count if source_class_count > 0 else 0

        is_poison = ratio < self.threshold
        print(f"The removed cluster is poisonous: {is_poison}. {ratio=}, {self.threshold=}, {label_count=}, {source_class_count=}")

    def analyze_training_data(self, source_class: int, dataloader: DataLoader, new_model, train_function, evaluate_function):
        activations = self._get_activations(dataloader)
        activations_ica_reduced = self._reduce_dimensions(activations)
        clusters_labels = self._get_kmeans_clusters(activations_ica_reduced)

        for num_cluster in range(self.kmeans_clusters):
            self._analyze_cluster_data(num_cluster, source_class, dataloader, new_model, clusters_labels, train_function, evaluate_function)
