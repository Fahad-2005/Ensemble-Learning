
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self): return self.value is not None

class DecisionTreeC45:
    def __init__(self, max_depth=5, min_samples_split=2, feature_subsampling=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subsampling = feature_subsampling  # Essential hook for Random Forests!
        self.root = None

    def _entropy(self, y):
        proportions = np.bincount(y) / (len(y) + 1e-9)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _best_split(self, X, y, feat_indices):
        best_gain_ratio = -1.0
        split_idx, split_thresh = None, None
        
        # Random Forest specific modification: Subsample features dynamically at each node split
        if self.feature_subsampling is not None:
            n_sub = min(len(feat_indices), self.feature_subsampling)
            feat_indices = np.random.choice(feat_indices, size=n_sub, replace=False)

        for feat in feat_indices:
            X_col = X[:, feat]
            thresholds = np.unique(X_col)
            if len(thresholds) > 20:
                thresholds = np.percentile(X_col, np.linspace(5, 95, 10))
                
            for thresh in thresholds:
                left_mask = X_col <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask): continue
                
                # C4.5 Split Evaluation Criteria
                p_entropy = self._entropy(y)
                n, n_l, n_r = len(y), np.sum(left_mask), np.sum(right_mask)
                c_entropy = (n_l/n)*self._entropy(y[left_mask]) + (n_r/n)*self._entropy(y[right_mask])
                info_gain = p_entropy - c_entropy
                
                split_info = -((n_l/n)*np.log2(n_l/n + 1e-9) + (n_r/n)*np.log2(n_r/n + 1e-9))
                gain_ratio = info_gain / (split_info + 1e-9)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio, split_idx, split_thresh = gain_ratio, feat, thresh
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split):
            return Node(value=np.bincount(y).argmax() if len(y) > 0 else 0)
            
        feat_indices = np.arange(n_feats)
        best_feat, best_thresh = self._best_split(X, y, feat_indices)
        if best_feat is None: return Node(value=np.bincount(y).argmax())
        
        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if node.is_leaf(): return node.value
        if x[node.feature] <= node.threshold: return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X): return np.array([self._predict_row(self.root, x) for x in X])