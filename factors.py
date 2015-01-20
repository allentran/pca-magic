import numpy as np
from sklearn import decomposition, linear_model, preprocessing
from scipy import interpolate

class Interpolater:

    def __init__(self, data, normalize=True, model='linear'):

        self.raw_data = data.copy()

        if normalize:
            for k in xrange(data.shape[1]):
                m = np.nanmean(self.raw_data[:,k])
                sd = np.nanstd(self.raw_data[:,k])
                self.raw_data[:,k] = (self.raw_data[:,k] - m)/float(sd)

        self.filled_data = self.raw_data.copy()

        self.model = 'linear'

    def get_factors(self, data, n_components, sample_range=None, usable=None):

        pca = decomposition.PCA()
        if (sample_range is not None) & (usable is not None):
            data = data[sample_range, :][:, usable]
        pca.fit(data)
        return pca.fit_transform(data)[:,:n_components+1]

    def interpolate(self, data, factors, n_factors=5):

        not_nan = ~np.isnan(data)
        test_data = data[not_nan]
        
        if not not_nan.all():
            if self.model == 'linear':
                clf = linear_model.LinearRegression()
                clf.fit(factors[not_nan,:n_factors], test_data)
                yhat = clf.predict(factors[:,:n_factors])
            data = yhat

        if np.max(data[~not_nan])>1000:
            import ipdb
            ipdb.set_trace()
           
        return data     

    def fit_iterative(self, init_size, n_components=10, reverse=True):

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



        def get_sample_range(idx):

            if reverse:
                sample_range = np.arange(idx, T)
            else:
                sample_range = np.arange(0, idx+1)
            return sample_range


        T = self.raw_data.shape[0]
        K = self.raw_data.shape[1]

        # find initial complete set of columns
        if reverse:
            sample_range = np.arange(T-init_size, T)
        else:
            sample_range = np.arange(0, init_size+1)

        valid = ~np.isnan(self.filled_data[sample_range]).any(axis=0)
        usable = [k for k in xrange(K) if valid[k]]

        self.factors = self.get_factors(self.raw_data.copy(), n_components, sample_range, usable)

        for idx in xrange(T-init_size-1, -1, -1):
            
            # find usable cols that are nan
            sample_range = get_sample_range(idx)
            nan_list = []
            if reverse:
                idx_lag = idx+1
            else:
                idx_lag = idx-1

            # front fill nan cols
            for k in usable: 
                if np.isnan(self.raw_data[idx, k]):
                    nan_list.append(k)
                    self.filled_data[idx,k] = self.filled_data[idx_lag, k]

            # get factors off front filled nans
            self.factors = self.get_factors(self.filled_data, n_components, sample_range, usable)
            
            # use factors and model to fill in 
            for k in nan_list:
                self.filled_data[sample_range, k] = self.interpolate(self.raw_data[sample_range, k].copy(), self.factors)

            new_usable = update_usable(idx, self.raw_data)
            usable += new_usable
            for k in new_usable:
                self.filled_data[sample_range, k] = self.interpolate(self.raw_data[sample_range, k].copy(), self.factors)
        
    def naive_interpolate(self, data): 

        def find_next_valid(idx=0):

            for ii in xrange(idx, len(data)):
                if not np.isnan(data[ii]):
                    return ii
            return None

        start = find_next_valid(0)
        end = find_next_valid(start+1)

        if start > 0:
            data[:start] = data[start]

        while end is not None:
            if end-start > 1:
                data[start+1:end] = (data[start]+data[end])/float(2)
            start = end
            end = find_next_valid(end+1)
            if end is None:
                data[start:] = data[start]

    def fit_recursive(self, n_components=10, min_obs=50, alpha=0.99):

        raw_data = self.raw_data[:, np.where(np.sum(~np.isnan(self.raw_data), axis=0) >= min_obs)[0]]

        T = raw_data.shape[0]
        K = raw_data.shape[1]

        self.filled_data = raw_data.copy()

        # initial interpolation
#        for k in xrange(K):
#            self.naive_interpolate(self.filled_data[:,k])
        self.filled_data[np.isnan(self.filled_data)] = 0

        self.factors = self.get_factors(self.filled_data, n_components)
        supnorm = 1692
        while supnorm > 1e-5:
            for k in xrange(K):
                self.filled_data[:, k] = self.interpolate(raw_data[:,k].copy(), self.factors)
            
            new_factors = (self.get_factors(self.filled_data, n_components)) + (1-alpha)*self.factors
            supnorm = np.max(np.abs(self.factors - new_factors))    
            import ipdb
            ipdb.set_trace()
            self.factors = new_factors
            print supnorm
            
            
