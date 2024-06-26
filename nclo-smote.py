import math
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_array
from scipy.spatial import distance_matrix



__author__ = 'taoll'



class NCLO_SMOTE(object):

    def __init__(self,
                 k = 5,
                 w_thres=90,
                 heat_thres = 10,
                 alpha = 0.5,
                 p_norm=2,
                 verbose=False):
        self.k_min = k
        self.w_thres = w_thres
        self.heat_thres = heat_thres
        self.alpha = alpha
        self.p_norm = p_norm
        self.verbose = verbose

    def fit(self,X,y):
        self.X = check_array(X)
        self.y = np.array(y)
        classes = np.unique(y)
        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.unique_classes_ = classes[indices]
        self.observation_dict = {c: X[y == c] for c in classes}
        self.maj_class_ = self.unique_classes_[0]
        n_max = max(sizes)
        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, max(sizes)))

    def fit_sample(self, X, y):
        self.fit(X, y)
        for i in range(1, len(self.observation_dict)):
            current_class = self.unique_classes_[i]
            self.n = len(self.observation_dict[0]) - len(self.observation_dict[current_class])
            reshape_points, reshape_labels = self.reshape_observations_dict()
            oversampled_points, oversampled_labels = self.generate_samples(reshape_points, reshape_labels, current_class)
            self.observation_dict = {cls: oversampled_points[oversampled_labels == cls] for cls in self.unique_classes_}
        reshape_points, reshape_labels = self.reshape_observations_dict()
        return reshape_points, reshape_labels



    def generate_samples(self, X, y, minority_class=None):
        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()
        self.n = len(majority_points) - len(minority_points)
        dm_maj = distance_matrix(minority_points, majority_points, self.p_norm)
        dm_min = distance_matrix(minority_points, minority_points, self.p_norm)
        clusters, labels_ = self.semi_cluster(minority_points,dm_maj,dm_min)
        self.compute_init_heat(majority_points, minority_points, clusters)
        majority_points = self.clear_overlap(majority_points, clusters)
        self.compute_weight(clusters)
        self.dc_min = distance_matrix(self.clusters_center, minority_points, self.p_norm)
        appended = []
        for i in range(len(clusters)):
            asc_min_index = np.argsort(self.dc_min[i])
            zero_index =[]
            for a,dx in enumerate(asc_min_index):
                if self.dc_min[i,dx] == 0.0:
                    zero_index.append(a)
            asc_min_index = np.delete(asc_min_index, zero_index)
            if self.k_min <= len(minority_points) - 1:
                knn_index = asc_min_index[0:self.k_min]
            else:
                knn_index = asc_min_index[0:]

            for _ in range(int(self.gi[i])):
                random_index = knn_index[np.random.choice(range(len(knn_index)), size=1, replace=False)]
                direction_vector = minority_points[random_index[0]] - self.clusters_center[i]
                direction_unit_vector = direction_vector / norm(direction_vector)
                new_data = self.clusters_center[i] + direction_unit_vector * np.random.rand() * self.radii[i]
                appended.append(new_data)
        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])
        return points, labels


    def compute_weight(self,clusters):
        min_weight = np.zeros(len(clusters))
        for i in range(len(clusters)):
            min_weight[i] = (1/len(clusters)) * self.local_overlap[i]
        max_weight = np.percentile(min_weight, q=self.w_thres)
        for i in range(len(clusters)):
            if min_weight[i] >= max_weight:
                min_weight[i] = max_weight
        weight_sum = np.sum(min_weight)
        for i in range(len(clusters)):
            min_weight[i] = min_weight[i] / weight_sum
        self.gi = np.rint(min_weight * self.n).astype(np.int32)

    def clear_overlap(self, majority_points, clusters):
        translations = np.zeros(majority_points.shape)
        if self.heat_thres is None:
            self.heat_thres = int(1/self.overlap_degree)
        heat_threshold = np.percentile(self.init_heat, q=self.heat_thres)
        for i in range(len(clusters)):
            num_maj_in_radius = 0
            asc_index = np.argsort(self.dc_maj[i])
            if self.init_heat[i] > heat_threshold:
                for j in range(1, len(asc_index)):
                    remain_heat = self.init_heat[i] * math.exp(-self.alpha * (self.dc_maj[i, asc_index[j]] - self.radii[i]))
                    num_maj_in_radius += 1
                    if remain_heat <= heat_threshold:
                        self.radii[i] = self.dc_maj[i, asc_index[j]]
                        break
            if num_maj_in_radius > 0:
                for j in range(num_maj_in_radius):
                        majority_point = majority_points[asc_index[j]]
                        d = np.sum(np.abs(majority_point - self.clusters_center[i]) ** self.p_norm) ** (1 / self.p_norm)

                        if d != 0.0:
                            translation = (self.radii[i] - d) / d * (majority_point - self.clusters_center[i])
                        else:
                            translation = self.radii[i] * (majority_point)
                        translations[asc_index[j]] = translations[asc_index[j]] + translation
        majority_points = majority_points.astype(np.float64)
        majority_points += translations

        return majority_points



    def compute_init_heat(self, majority_points, minority_points, clusters):
        self.clusters_center = np.zeros((len(clusters),minority_points.shape[1]))
        for i in range(len(clusters)):
            if len(clusters[i]) > 1:
                self.clusters_center[i] = np.mean(minority_points[clusters[i]], axis=0)
            else:
                self.clusters_center[i] = minority_points[clusters[i]]
        self.dc_maj = distance_matrix(self.clusters_center, majority_points, self.p_norm)
        self.radii = np.min(self.dc_maj, axis=1)
        maj_center = np.mean(majority_points, axis=0)
        min_center = np.mean(minority_points, axis=0)
        self.overlap_degree = len(clusters)/len(minority_points)
        self.init_heat = np.zeros(len(clusters))
        self.local_overlap = np.zeros(len(clusters))
        for i in range(len(clusters)):
            dist_maj_center = np.linalg.norm(self.clusters_center[i] - maj_center)
            dist_min_center = np.linalg.norm(self.clusters_center[i] - min_center)
            self.local_overlap[i] = (dist_min_center/dist_maj_center)
            self.init_heat[i] = self.overlap_degree * (1/len(clusters)) * self.local_overlap[i]

    def semi_cluster(self,minority_points, dm_maj, dm_min):
        min_maj_values = np.min(dm_maj, axis=1)
        dm_min = dm_min.copy()
        minr_num, minc_num = np.shape(minority_points)
        for idx in range(len(dm_min)):
            dm_min[idx, idx] = np.inf
        clusters = [[i] for i in range(minr_num)]
        while not (len(clusters)== 1 or np.all(np.isinf(dm_min))):
            min_dist = math.inf
            merge_clusters = None
            for i in range(len(dm_min) - 1):
                for j in range(i + 1, len(dm_min)):
                    if dm_min[i,j] < min_dist:
                        min_dist = dm_min[i,j]
                        merge_clusters = (i,j)
            part1, part2 = merge_clusters
            if min_maj_values[part1] < min_dist or min_maj_values[part2] < min_dist:
                dm_min[part1, part2] = np.inf
                dm_min[part2, part1] = np.inf
            else:
                clusters[part1].extend(clusters[part2])
                del clusters[part2]
                dm_min[part1] = np.min(
                    np.vstack([dm_min[part1], dm_min[part2]]), axis=0
                )
                dm_min[:, part1] = dm_min[part1]
                dm_min = np.delete(dm_min, part2, axis=0)
                dm_min = np.delete(dm_min, part2, axis=1)
                dm_min[part1, part1] = np.inf
                if min_maj_values[part1] < min_maj_values[part2]:
                    min_maj_values[part1] = min_maj_values[part2]
                min_maj_values = np.delete(min_maj_values, part2)
        labels_ = np.zeros(minr_num, dtype=int)
        for i, cluster in enumerate(clusters):
            labels_[cluster] = i
        return clusters,labels_

    def reshape_observations_dict(self):
        reshape_points = []
        reshape_labels = []
        for cls in self.observation_dict.keys():
            if len(self.observation_dict[cls]) > 0:
                reshape_points.append(self.observation_dict[cls])
                reshape_labels.append(np.tile([cls], len(self.observation_dict[cls])))
        reshape_points = np.concatenate(reshape_points)
        reshape_labels = np.concatenate(reshape_labels)

        return reshape_points, reshape_labels



