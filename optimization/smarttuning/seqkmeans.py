from __future__ import  annotations
from typing import Union
import numpy as np
from collections import Counter
from sampler import Metric
import pandas as pd
from Levenshtein import StringMatcher
import logging
import uuid
import copy

from scipy.spatial.distance import cdist as ed
from scipy import special

import config

logger = logging.getLogger(config.KMEANS_LOGGER)
logger.setLevel(logging.DEBUG)


def __merge_data__(data1:pd.Series, data2:pd.Series) -> (pd.Series, pd.Series, pd.Index):
    merged = pd.merge(data1, data2, how='outer', left_index=True, right_index=True)
    merged = merged.replace(float('NaN'), 0)
    columns = [column for column in merged.columns[:2]]

    return merged[columns[0]], merged[columns[1]], merged.index

def __grouping_rows__(data:pd.Series, threshold) -> pd.Series:
    return __hist_reduce__(__compare__(data, threshold))

def __fuzzy_string_comparation__(u:str, v:str, threshold:float):
    """
        compare two strings u, v using Levenshtein distance
        https://en.wikipedia.org/wiki/Levenshtein_distance

        :return how many changes (%) are necessary to transform u into v
    """
    diff = StringMatcher.distance(u, v)
    return diff / len(u) < threshold

def __distance__(u:pd.Series, v:pd.Series, distance=config.DISTANCE_METHOD.lower()) -> float:
    SQRT2 = np.sqrt(2)

    _distance = {
        'hellinger': lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2,
        'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        'euclidean': lambda a, b: np.linalg.norm(a-b)
    }

    return _distance[distance](u, v) or 0

def __compare__(histograms:pd.Series, threshold:int):
    from collections import defaultdict
    workflows_group = defaultdict(set)
    memory = set()

    class Item:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __eq__(self, other):
            return self.key == other.key

        def __hash__(self):
            return hash(self.key)

        def __repr__(self):
            return f'({self.key}:{self.value})'

        def acc(self, value):
            self.value += value

    if len(histograms) > 0:
        # groups similar urls
        for hist1, i in histograms.items():
            for hist2, j in histograms.items():
                if __fuzzy_string_comparation__(hist1, hist2, threshold):
                    __group__(Item(hist1, i), Item(hist2, j), workflows_group, memory)
    return workflows_group

# TODO: optimize this in the future
def __group__(a, b, table, memory):
    if a not in memory and b not in memory:
        table[a].add(b)
    elif a in memory and b not in memory:
        if a in table:
            table[a].add(b)
        else:
            return __group__(b, a, table, memory)
    elif a not in memory and b in memory:
        for key, value in table.items():
            if b in value:
                value.add(a)
                break
    memory.add(a)
    memory.add(b)


def __hist_reduce__(table):
    data = {'path': [], 'value': []}

    for key, items in table.items():
        key.value = 0
        for value in items:
            key.acc(value.value)
        data['path'].append(key.key)
        data['value'].append(key.value)

    return pd.Series(data = data['value'], index=data['path']).sort_index()


# implementation note
# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
#
class Container:
    def __init__(self, label, content:pd.Series=None, metric=None,
                 similarity_threshold=config.URL_SIMILARITY_THRESHOLD):

        self.label = label
        self.content:pd.Series = __grouping_rows__(content, similarity_threshold)
        self.content.name = label

        self.metric = metric
        if self.metric is None:
            self.metric = Metric(cpu=0, memory=0, throughput=0, process_time=0, errors=0)

        self.configuration = None

        self.start = None
        self.end = None

        self.classification = None
        self.hits = 0

    def __str__(self):
        return f'label:{self.label}, classification:{self.classification.id if self.classification else ""}, config:{self.configuration}'

    def __lt__(self, other):
        return self.metric < other.metric

    def serialize(self):
        container_dict = copy.deepcopy(self.__dict__)
        container_dict['content_labels'] = self.content.index.to_list()
        container_dict['content'] = self.content.to_list()
        container_dict['metric'] = self.metric.serialize()

        if isinstance(self.classification, Cluster):
            container_dict['classification'] = self.classification.id
        elif isinstance(self.classification, str):
            container_dict['classification'] = self.classification
        return container_dict

    def distance(self, other:Union[pd.Series, Container]):
        if isinstance(other, Container):
            other = other.content
        u, v, _ = __merge_data__(self.content, other)
        return __distance__(u, v)

class Cluster:
    # todo: merge clusters they have save name
    def __init__(self, container:Container=None):
        self.id = str(uuid.uuid1())
        self.hits = 0
        self.len = 0
        self.counter = Counter()
        self.center = pd.Series(name=self.id, dtype=np.float)

        if container:
            self.add(container)

    def __str__(self):
        return f'[{len(self):04d}] {self.name()}: {len(self.counter)}'

    def __len__(self):
        return self.len

    def __eq__(self, other):
        return self.id == other.id

    def name(self):
        if self.most_common():
            return self.id + ' ' + self.most_common()[0][0]
        else:
            return self.id

    def most_common(self):
        return self.counter.most_common(1)

    def add(self, container:Container):
        self.len += 1
        self.counter[container.label] += 1

        v, u, index = __merge_data__(container.content, self.center)

        center = v + (u - v) * (1/len(self))

        self.center = pd.Series(data=center, index=index, name=self.id)

    def centroid(self):
        return self.center

    def inc(self):
        self.hits += 1

class KmeansContext:
    def __init__(self, k, cluster_type=Cluster):
        self.closest_cluster = None
        self.most_common_cluster = None
        self.most_common = 0
        self.min_distance = float('inf')
        self.clusters = []
        self.k = k
        self.cluster_type = cluster_type

    def cluster_by_id(self, id):
        for cluster in self.clusters:
            if id == cluster.id:
                return cluster
        # return self.clusters[np.random.randint(0, len(self.clusters))]
        logging.warning('returning None cluster')
        return None

    def cluster(self, sample:Container)->(Cluster, int):
        assert isinstance(sample, Container)
        if len(self.clusters) < self.k:
            self.clusters.append(self.cluster_type(sample))

        self.closest_cluster = self.clusters[0]
        self.most_common_cluster = self.clusters[0]
        self.most_common = 0
        self.min_distance = float('inf')

        for cluster in self.clusters:
            distance = sample.distance(cluster.centroid())
            if distance < self.min_distance:
                self.min_distance = distance
                self.closest_cluster = cluster

            if len(cluster.most_common()):
                frequency = cluster.most_common()[0][1]
                label = cluster.most_common()[0][0]

                if label == sample.label and frequency > self.most_common:
                    self.most_common = frequency
                    self.most_common_cluster = cluster

        self.closest_cluster.add(sample)
        best_cluster:Cluster = self.closest_cluster

        best_cluster.inc()

        return best_cluster, best_cluster.hits

class GPRNP(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, check_numerics=True, debug=False):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.check_numerics = check_numerics
        self.debug = debug
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.items()):
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPRNP")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPRNP")

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=0.01):
        self._reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        if X_train.ndim != 2 or y_train.ndim != 2:
            raise Exception("X_train or y_train should have 2 dimensions! X_dim:{}, y_dim:{}"
                            .format(X_train.ndim, y_train.ndim))
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1
        K = self.magnitude * np.exp(-ed(self.X_train, self.X_train) / self.length_scale) \
            + np.diag(ridge)
        K_inv = np.linalg.inv(K)
        self.K = K
        self.K_inv = K_inv
        self.y_best = np.min(y_train)
        return self

    def predict(self, X_test):
        self.check_fitted()
        if X_test.ndim != 2:
            raise Exception("X_test should have 2 dimensions! X_dim:{}"
                            .format(X_test.ndim))
        X_test = np.float32(GPRNP.check_array(X_test))
        test_size = X_test.shape[0]
        arr_offset = 0
        length_scale = self.length_scale
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        eips = np.zeros([test_size, 1])
        while arr_offset < test_size:
            if arr_offset + self.batch_size_ > test_size:
                end_offset = test_size
            else:
                end_offset = arr_offset + self.batch_size_
            xt_ = X_test[arr_offset:end_offset]
            K2 = self.magnitude * np.exp(-ed(self.X_train, xt_) / length_scale)
            K3 = self.magnitude * np.exp(-ed(xt_, xt_) / length_scale)
            K2_trans = np.transpose(K2)
            yhat = np.matmul(K2_trans, np.matmul(self.K_inv, self.y_train))
            sigma = np.sqrt(np.diag(K3 - np.matmul(K2_trans, np.matmul(self.K_inv, K2)))) \
                .reshape(xt_.shape[0], 1)
            u = (self.y_best - yhat) / sigma
            phi1 = 0.5 * special.erf(u / np.sqrt(2.0)) + 0.5
            phi2 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(np.square(u) * (-0.5))
            eip = sigma * (u * phi1 + phi2)
            yhats[arr_offset:end_offset] = yhat
            sigmas[arr_offset:end_offset] = sigma
            eips[arr_offset:end_offset] = eip
            arr_offset = end_offset
        GPRNP.check_output(yhats)
        GPRNP.check_output(sigmas)
        return GPRResult(yhats, sigmas)

    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "K": self.K,
                "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        return self

class GPRResult(object):

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas
