
import numpy as np
from decision_tree_base import DecisionTreeC45

class RandomForestClassifierScratch:
    def __init__(self, n_estimators=15, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.feature_usage_counts = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Subsampling ratio boundary rule for random forests: sqrt(d)
        n_subspace = int(np.floor(np.sqrt(n_features)))
        
        self.estimators = []
        self.feature_usage_counts = np.zeros(n_features)
        
        # Keep track of which samples were out-of-bag for each tree
        oob_predictions = np.zeros((n_samples, self.n_estimators))
        oob_predictions.fill(np.nan)

        for i in range(self.n_estimators):
            # Row Bootstrap Sample
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b, y_b = X[boot_idx], y[boot_idx]
            
            # Identify OOB indices for this specific estimator instance
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_indices = np.where(oob_mask)[0]

            # Fit Tree using the feature_subsampling parameter
            tree = DecisionTreeC45(max_depth=self.max_depth, feature_subsampling=n_subspace)
            tree.fit(X_b, y_b)
            self.estimators.append(tree)
            
            # Populate OOB prediction metrics
            if len(oob_indices) > 0:
                oob_predictions[oob_indices, i] = tree.predict(X[oob_indices])

        # Core computation loop for Part C: Manual OOB Error Evaluation
        oob_votes = []
        valid_oob_samples = 0
        correct_oob_samples = 0
        
        for idx in range(n_samples):
            row_votes = oob_predictions[idx, ~np.isnan(oob_predictions[idx])]
            if len(row_votes) > 0:
                final_vote = np.bincount(row_votes.astype(int)).argmax()
                valid_oob_samples += 1
                if final_vote == y[idx]:
                    correct_oob_samples += 1
                    
        self.oob_score_ = correct_oob_samples / (valid_oob_samples + 1e-9)
        return self.oob_score_

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.estimators])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)