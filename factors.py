import os

import numpy as np
from sklearn import decomposition, linear_model, preprocessing
import pandas as pd

class Interpolater:

    def __init__(self, data, normalize=False, reverse=True, model='linear'):

        self.raw_data = data.copy()

        if normalize:
            for k in xrange(data.shape[1]):
                m = np.nanmean(self.raw_data[:,k])
                sd = np.nanstd(self.raw_data[:,k])
                self.raw_data[:,k] = (self.raw_data[:,k] - m)/float(sd)

        self.filled_data = self.raw_data.copy()

        self.reverse = True
        self.model = 'linear'

    def fit(self, init_size, var_explained=0.9):

        def get_factors(sample_range, data):

            pca = decomposition.PCA()
            pca.fit(data[sample_range, :][:,usable])
            explained_var = pca.explained_variance_ratio_.cumsum()
            first_idx = np.where(explained_var>=var_explained)[0]
            if len(first_idx)>0:
                pca = pca.set_params(n_components=min(first_idx))
            pca = pca.set_params(n_components=10)
            return pca.fit_transform(data[sample_range, :][:,usable])

        def update_usable(idx, data):

            # interpolate newly usable

            new_idxs = []
            sample_range = get_sample_range(idx)

            for k in xrange(K):
                if k in usable:
                    continue
                if is_usable(sample_range, k, data):
                    new_idxs.append(k)

            return new_idxs

        def is_usable(sample_range, k, data):

            not_nan = ~np.isnan(data[sample_range, k])
            return not_nan.sum() >= init_size

        def interpolate(data, factors):

            not_nan = ~np.isnan(data)
            test_data = data[not_nan]
            
            if not not_nan.all():
                if self.model == 'linear':
                    clf = linear_model.LinearRegression()
                    try:
                        clf.fit(factors[not_nan,:], test_data)
                    except ZeroDivisionError:
                        pass
                    yhat = clf.predict(factors)
                data[~not_nan] = yhat[~not_nan]
               
            return data

        def get_sample_range(idx):

            if self.reverse:
                sample_range = np.arange(idx, T)
            else:
                sample_range = np.arange(0, idx+1)
            return sample_range


        T = self.raw_data.shape[0]
        K = self.raw_data.shape[1]

        # find initial complete set of columns
        if self.reverse:
            sample_range = np.arange(T-init_size, T)
        else:
            sample_range = np.arange(0, init_size+1)

        valid = ~np.isnan(self.filled_data[sample_range]).any(axis=0)
        usable = [k for k in xrange(K) if valid[k]]

        self.factors = get_factors(sample_range, self.raw_data.copy())

        for idx in xrange(T-init_size-1, -1, -1):
            
            # find usable cols that are nan
            sample_range = get_sample_range(idx)
            nan_list = []
            if self.reverse:
                idx_lag = idx+1
            else:
                idx_lag = idx-1

            # front fill nan cols
            for k in usable: 
                if np.isnan(self.raw_data[sample_range, k]).any():
                    nan_list.append(k)
                    self.filled_data[idx,k] = self.filled_data[idx_lag, k]

            # get factors off front filled nans
            self.factors = get_factors(sample_range, self.filled_data)
            
            # use factors and model to fill in 
            for k in nan_list:
                self.filled_data[sample_range, k] = interpolate(self.raw_data[sample_range, k].copy(), self.factors)

            new_usable = update_usable(idx, self.raw_data)
            usable += new_usable
            for k in new_usable:
                self.filled_data[sample_range, k] = interpolate(self.raw_data[sample_range, k].copy(), self.factors)
            print idx
