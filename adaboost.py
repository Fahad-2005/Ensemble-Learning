
import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = 0
        else:
            predictions[X_column > self.threshold] = 0
        return predictions

class AdaBoostClassifierScratch:
    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Step 1: Initialize sample weights uniformly
        w = np.ones(n_samples) / n_samples
        self.estimators = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            # Search the entire feature space to find the most optimal split stump
            for feat_i in range(n_features):
                X_column = X[:, feat_i]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        p = np.ones(n_samples)
                        if polarity == 1:
                            p[X_column < threshold] = 0
                        else:
                            p[X_column > threshold] = 0
                        
                        # Misclassification Error weighted by the sample weight vector
                        error = np.sum(w[y != p])
                        
                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.feature_idx = feat_i
                            stump.threshold = threshold

            # Step 2: Compute alpha (voting weight of the stump)
            # Clip error to avoid divide-by-zero errors
            min_error = np.clip(min_error, 1e-10, 1.0 - 1e-10)
            stump.alpha = 0.5 * np.log((1.0 - min_error) / min_error)

            # Step 3: Update sample weights sequentially
            predictions = stump.predict(X)
            # If prediction matches y, exponent factor is negative; otherwise positive
            mismatch_sign = np.where(y == predictions, -1, 1)
            w *= np.exp(stump.alpha * mismatch_sign)
            
            # Normalize sample weights so they sum up to 1
            w /= np.sum(w)
            self.estimators.append(stump)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * (1 if val == 1 else -1) for stump in self.estimators for val in stump.predict(X)])
        stump_preds = stump_preds.reshape(len(self.estimators), X.shape[0])
        raw_sum = np.sum(stump_preds, axis=0)
        return np.where(raw_sum >= 0, 1, 0)