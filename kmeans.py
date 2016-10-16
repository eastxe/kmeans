#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy as sp


class Kmeans():
    def __init__(self, dist='Euclidean', n_class=2):
        self.k = 0
        self.n_data = 0
        self.dist = dist
        self.n_class = n_class

    def run(self, features):
        self.features = np.array(features)
        print self.features
        self.mean = np.random.randint(0, self.features.max(),
                (self.n_class, len(features[0])))
        self.label = np.random.randint(0, self.n_class, len(self.features))
        old_mean = np.ones((self.n_class, len(features[0])))
        while not self.diff(old_mean):
            old_mean = self.mean
            self.update()
            self.calc_mean()
        print 'end'
        print self.mean
        print 'label: ', self.label

    def update(self):
        for i, feature in enumerate(self.features):
            dists = []
            for center in self.mean:
                dists.append(self.distance(feature, center))
            label = dists.index(min(dists))
            self.label[i] = label

    def calc_mean(self):
        """
            To calculate a new cluster average points
        """
        sum_features = np.zeros((self.n_class, len(self.features[0])))
        num_features = np.zeros(self.n_class)
        for i, clas in enumerate(self.label):
            sum_features[clas] += self.features[i]
            num_features[clas] += 1
        try:
            # TODO: refactaring
            temp = []
            for i, ary in enumerate(sum_features):
                temp2 = []
                for j, value in enumerate(ary):
                    temp2.append(value / num_features[i])
                temp.append(temp2)
            a = np.array((temp))
            return a

        except ZeroDivisionError:
            raise ZeroDivisionError('0 divison error')

    def diff(self, old_mean):
        for difference in self.mean - old_mean:
            if (abs(difference) > 0.0001).any():
                return False
        return True

    def distance(self, vec, mean):
        if self.dist == 'Euclidean':
            return np.linalg.norm(vec - mean)
        elif self.dist == 'cosine':
            return sp.spatial.distance.cosine(vec, mean)
        else:
            raise ValueError('%s isunknown way to calc distance' % self.dist)


if __name__ == '__main__':
    I = Kmeans()
    ary = [[0, 1], [0, 0], [10, 10], [11, 11]]
    I.run(ary)
