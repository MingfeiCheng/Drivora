import joblib
import numpy as np

from scipy import interpolate
from sklearn.cluster import KMeans

def mean_change(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    diffs = np.diff(x)
    return np.mean(diffs)

def mean_abs_change(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    diffs = np.diff(x)
    return np.mean(np.abs(diffs))

def c3(x: np.ndarray, lag: int = 1) -> float:
    # Third order autocorrelation
    if len(x) < 3 * lag:
        return 0.0
    return np.mean((x[:-2*lag] * x[lag:-lag] * x[2*lag:]))

def cid_ce(x: np.ndarray, normalize: bool = True) -> float:
    # Complexity-invariant distance (CE) approximation
    if len(x) < 2:
        return 0.0
    diffs = np.diff(x)
    ce = np.sqrt(np.sum(diffs ** 2))
    if normalize:
        ce /= len(diffs)
    return ce

def static_calculator(X: np.ndarray) -> list:
    X_feature = []
    n_attr = X.shape[1]
    for i in range(n_attr):
        xi = X[:, i]

        mean = np.mean(xi)
        minimum = np.min(xi)
        maximum = np.max(xi)
        var = np.var(xi)
        m_change = mean_change(xi)
        m_abs_change = mean_abs_change(xi)
        c3_val = c3(xi, lag=1)
        cid_val = cid_ce(xi, normalize=True)

        features = [mean, var, minimum, maximum, m_change, m_abs_change, c3_val, cid_val]
        X_feature.extend(features)

    return X_feature

class FeatureNet(object):

    def __init__(self, window_size=1):
        self.resample_frequency = 0
        self.window_size = window_size # unit is second (s)
        self.local_feature_extractor = None

    @staticmethod
    def input_resample(xs, ts, resample='linear', sample_frequency=0.1):
        # x: [t, m], t: [t]
        x = np.array(xs)
        resample_axis = np.arange(ts[0], ts[-1], sample_frequency)
        new_x = []
        for i in range(0, x.shape[1]):
            x_i = x[:, i] # [t]
            f_i = interpolate.interp1d(ts, x_i, kind=resample)
            new_x_i = f_i(resample_axis) # [t]
            new_x_i = np.append(new_x_i, x_i[-1])
            new_x.append(new_x_i)
        new_x = np.array(new_x)
        new_x = new_x.T
        # new_x: [t, m]
        return new_x

    def forward(self, x, resample='linear'):
        """
        x is the list of scene observations
        """
        # use attributes: heading, speed, acceleration
        # aims to assign the time stamp for feature extraction!!!
        x_behavior_vector = np.array(x)

        time_size = x_behavior_vector.shape[0]
        if time_size < self.window_size:
            last_element = x_behavior_vector[-1:, :]
            for _ in range(self.window_size - time_size):
                x_behavior_vector = np.concatenate([x_behavior_vector, last_element], axis=0)

        y = []
        for i in range(time_size - self.window_size + 1):
            x_segment = x_behavior_vector[i:i+self.window_size]
            x_feature = static_calculator(x_segment)
            y.append(x_feature)

        return np.array(y)

class ClusterModelBehavior(object):
    def __init__(self, cluster_num):
        """
        Initial cluster number
        """
        self.cluster_model = KMeans(cluster_num)
        self.cluster_data = None

    def search(self, v):
        """
        @param: v is the query feature
        """
        # v represents the behaviors of a single case
        # @format is numpy with shape (n, 64)
        # @output is numpy with shape (n, )
        cls_labels = self.cluster_model.predict(v)
        # nearest_node = self.AI.get_nns_by_vector(v, 1, include_distances=True)
        # label(node id) & distance
        return cls_labels

    def update(self, v):
        """
        Need to change to load all corpus and re-cluster
        """
        v[np.isnan(v)] = 0
        # Step1: add new behavior data @format is numpy with shape (n, 64)
        if self.cluster_data is None:
            self.cluster_data = v
        else:
            self.cluster_data = np.concatenate([self.cluster_data, v], axis=0)
        # Step2: retrain kmeans model.
        y = self.cluster_model.fit_predict(self.cluster_data) # shape (n, )
        return y

    def get_centers(self):
        return self.cluster_model.cluster_centers_
    
    def save(self, path: str):
        """
        Save the whole object (cluster model + cluster data)
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load the whole object back
        """
        return joblib.load(path)
    
class CoverageModel(object):

    def __init__(
        self, 
        window_size, 
        cluster_num, 
        threshold_coverage,
        use_dynamic_threshold=False
    ):

        self.coverage_centers = []
        self.coverage_centers_index = []
        self.coverage_centers_pointer = 0

        self.window_size = window_size
        self.cluster_num = cluster_num
        self.threshold_coverage = threshold_coverage

        self.dynamic_threshold = np.inf

        self.feature_layer = FeatureNet(window_size)
        self.cluster_layer_behavior = ClusterModelBehavior(self.cluster_num)
        
        self.use_dynamic_threshold = use_dynamic_threshold
        
    def save(self, path: str):
        """Save the whole CoverageModel object."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Load the CoverageModel object back."""
        return joblib.load(path)

    def _extract_feature(self, x, resample='linear'):
        y_behavior = self.feature_layer.forward(x, resample)
        return y_behavior

    def initialize(self, X: list):
        """
        X_behavior: list [item1, item2, ..., itemn]
            itemi : array [[x1...], [x2...]]
        X_trace: list [item1, item2, ..., itemn]
            itemi: list: [(x1, y1), (x2, y2), ..., (xn, yn)]
        """
        X_behavior = []
        for x in X:
            X_behavior.append(self._extract_feature(x, 'linear'))

        # behavior model
        buffer_feature = None
        for i in range(len(X_behavior)):
            x = X_behavior[i] # shape (n, 64)
            if buffer_feature is None:
                buffer_feature = x
            else:
                buffer_feature = np.concatenate([buffer_feature, x], axis=0)

            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]

        # initial train
        y = self.cluster_layer_behavior.update(buffer_feature) # n x 64
        self.update(y)

    def update(self, y):
        """
        y is the class labels of all cases
         shape is (n, ), the cluster label sequence.
        """
        self.coverage_centers = []
        for item in self.coverage_centers_index:
            start_index = item[0]
            end_index = item[1]
            y_i = y[start_index:end_index]
            self.coverage_centers.append(y_i)
            
        if self.use_dynamic_threshold:
            self._update_threshold()

    def _update_threshold(self):
        pattern_num = len(self.coverage_centers)
        distance_matrix = np.zeros((pattern_num, pattern_num))
        for i in range(pattern_num):
            distance_matrix[i][i] = 1000
            for j in range(i + 1, pattern_num):
                tmp_distance = self._compute_distance_behavior_states(self.coverage_centers[i], self.coverage_centers[j])
                distance_matrix[i][j] = tmp_distance
                distance_matrix[j][i] = tmp_distance

        pattern_min_distance = []
        for i in range(pattern_num):
            pattern_i_min = np.min(distance_matrix[i])
            pattern_min_distance.append(pattern_i_min)
        pattern_min_distance = np.array(pattern_min_distance)
        self.dynamic_threshold = np.mean(pattern_min_distance)

    def feedback_coverage_behavior(self, x):

        # x = self._extract_feature(seed, 'linear')

        y_behavior = self.cluster_layer_behavior.search(x)
        find_new_coverage = False
        min_feedback = np.inf
        for i in range(len(self.coverage_centers)):
            cov_feedback = self._compute_distance_behavior_states(y_behavior, self.coverage_centers[i])
            if cov_feedback < min_feedback:
                min_feedback = cov_feedback
        # if min_feedback > min(self.dynamic_threshold, self.threshold_coverage):
        if min_feedback > self.threshold_coverage:

            find_new_coverage = True
            # if no_pass: # ignore whether the seed is fail or pass
            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]
            # update behavior model (kmeans)
            y = self.cluster_layer_behavior.update(x)
            # update existing centers
            self.update(y)

        return find_new_coverage, min_feedback, y_behavior

    @staticmethod
    def _compute_distance_behavior_states(y1, y2):
        """
        y1 is a list
        """
        # y is numpy
        y1_length = len(y1)
        y2_length = len(y2)

        coverage_score = abs(y1_length - y2_length)

        common_length = min(y1_length, y2_length)
        y1_common = y1[:common_length]
        y2_common = y2[:common_length]
        for i in range(common_length):
            y1_e = y1_common[i]
            y2_e = y2_common[i]
            if y1_e == y2_e:
                continue
            else:
                coverage_score += 1

        coverage_score /= float(max(y1_length, y2_length))

        return coverage_score

    def get_centers(self):
        return self.coverage_centers