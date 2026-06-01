
import numpy as np
from decision_tree_base import DecisionTreeC45

class BaggingClassifierScratch:
    def __init__(self, n_estimators=10, max_depth=8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def draw_bootstrap_sample(self, X, y):
        """Part A: Implement Bootstrap Sampling Manually"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Track Out-of-Bag indices for visual evaluation
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        oob_indices = np.where(oob_mask)[0]
        
        return X[indices], y[indices], indices, oob_indices

    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            X_b, y_b, _, _ = self.draw_bootstrap_sample(X, y)
            tree = DecisionTreeC45(max_depth=self.max_depth)
            tree.fit(X_b, y_b)
            self.estimators.append(tree)

    def predict(self, X):
        # Accumulate predictions across all baseline trees
        tree_preds = np.array([tree.predict(X) for tree in self.estimators])
        # Majority Voting rule application
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        return majority_vote