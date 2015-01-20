import numpy as np

# terminology
# XW = data
# X: n by k, W: k by d, data: n by d  

class ProbPCA:

    def __init__(self, data):

        self.raw_data = data

    def trim_normalize(self, data, min_obs=10):

        any_valid = (~np.isnan(self.data)).any(axis=1)
        data = data[any_valid, :]

        any_valid = np.sum((~np.isnan(data)),axis=0) > min_obs
        data = data[:, any_valid]
    
        for idx in xrange(data.shape[1]):
            data[:, idx] = (data[:, idx] - np.nanmean(data[:, idx]))/np.nanstd(data[:, idx])
        
        return data

    def update_xy(self, W, fill_y):

        if fill_y:
            replace = np.isnan(self.data)
            y0 = self.data.copy()
            y0[replace] = 0
            
            supnorm = 1692
            while supnorm > 1e-2:
                X = np.dot(np.linalg.inv(np.dot(W.T, W)), np.dot(W, y0.T))
                y1 = np.dot(W,X)
                supnorm = np.max(np.abs(y0[replace]-y1[replace]))
                print supnorm
                y0[replace] = 0.1*y1[replace].copy() + 0.9*y0[replace]
        else:
            X = np.dot(np.linalg.inv(np.dot(W.T, W)), np.dot(W.T, self.data))
            y0 = self.data

        return X, y0

    def _update_covx(self, var_y, W):

        def calc_wmat():
            
            wjs = np.zeros((self.n_c, self.n_c, self.n_d))
            for idx_i in xrange(self.n_d):
                wjs[:, :, idx_i] = W[idx_i, :].T * W[idx_i, :]
            return wjs
            
        cov_x = np.zeros((self.n_c, self.n_c, self.n))
        w_mat = calc_wmat()
        for idx_j in xrange(self.n):
            ww_sum = np.sum(w_mat[:, :, self.O_j[idx_j]], axis=3).squeeze()
            cov_x[:, :, idx_j] = var_y * np.linalg.inv(var_y*np.eye(self.n_c) + ww_sum)

        return cov_x

    def _update_meanx(self, var_y, W, m, cov_x):

        def calc_diff(): # w_i * (y_ij - m_i)
            diffs = np.zeros((self.n_c, 1, self.n_d, self.n))
            for idx_i in xrange(self.n_d):
                for idx_j in xrange(self.n):
                    diffs[:, :, idx_i, idx_j] = W[idx_i, :].T * (self.data[idx_i, idx_j] - m[idx_i]) 
            return diffs

        mean_x = np.zeros((self.n_c, 1, self.n))
        diffs = calc_diff()
        for idx_j in xrange(self.n):
            _sum = np.sum(np.squeeze(diffs[:, :, self.O_j[idx_j], idx_j], axis=2), axis=2) 
            mean_x[:, :, idx_j] = (1/var_y) * np.dot(cov_x[:, :, idx_j], _sum)
        return mean_x

    def _update_meany(self, W, mean_x):

        def calc_diff(): # y_ij - w_i*x_bar_j
            diffs = np.zeros((self.n_d, self.n))
            for idx_i in xrange(self.n_d):
                for idx_j in xrange(self.n):
                    diffs[idx_i, idx_j] = self.data[idx_i, idx_j] - np.dot(W[idx_i, :], mean_x[:, :, idx_j])
            return diffs

        mean_y = np.zeros((self.n_d, 1))
        diffs = calc_diff()
        for idx_i in xrange(self.n_d):
            mean_y[idx_i,0] = np.mean(diffs[idx_i, self.O_i[idx_i]])
        return mean_y

    def _update_W(self, cov_x, mean_x, mean_y):

        def calc_diffs():

            diffs = np.zeros((self.n_c, 1, self.n_d, self.n))
            for idx_i in xrange(self.n_d):
                for idx_j in xrange(self.n):
                    diffs[:, :, idx_i, idx_j] = mean_x[:, :, idx_j] * (self.data[idx_i, idx_j] - mean_y[idx_i])
            return diffs

        def calc_prodx():

            prodx = np.zeros((self.n_c, self.n_c, self.n))
            for idx_j in xrange(self.n):
                prodx[:, :, idx_j] = np.dot(mean_x[:, :, idx_j], mean_x[:, :, idx_j].T) + cov_x[:, :, idx_j]
            return prodx

        W = np.zeros((self.n_d, self.n_c))
        diffs = calc_diffs()
        prods = calc_prodx()

        for idx_i in xrange(self.n_d):
            W[idx_i, :] = np.dot(np.linalg.inv(np.sum(prods[:, :, self.O_i[idx_i]].squeeze(), axis=2)), np.sum(diffs[:, :, idx_i, self.O_i[idx_i]].squeeze(axis=2), axis=2)).ravel()
        return W

    def _update_vary(self, W, cov_x, mean_x, mean_y):

        var_ys = np.zeros((self.n_d, self.n))
        for idx_i in xrange(self.n_d):
            for idx_j in xrange(self.n):
                square_term = self.data[idx_i, idx_j] - np.dot(W[idx_i, :], mean_x[:, :, idx_j]) - mean_y[idx_i]
                quad_term = np.dot(np.dot(W[idx_i, :], cov_x[:, :, idx_j]), W[idx_i,:].T)
                var_ys[idx_i, idx_j] = square_term**2 + quad_term
        
        var_y = np.mean(var_ys[self.O])
        return var_y

    def fit_transform(self, n_components=5): 

        self.data = self.raw_data.copy()
        self.data = self.trim_normalize(self.data).T
        self.n_d = self.data.shape[0]
        self.n = self.data.shape[1]
        self.n_c = n_components

        # dimensions(time) observed of series j
        self.O_j = [np.where(~np.isnan(self.data)[:, j]) for j in xrange(self.n)]
        self.O_i = [np.where(~np.isnan(self.data)[i, :]) for i in xrange(self.n_d)]
        self.O = ~np.isnan(self.data)

        missing_values = np.isnan(self.data).any()
        self._fit_transform_missing()

    def _fit_transform_missing(self):

        supnorm = 1692
        W0 = np.random.normal(size=[self.n_d, self.n_c])
        var_y = 1e-2
        m = np.nanmean(self.data, axis=1)

        while supnorm > 1e-2:
            cov_x = self._update_covx(var_y, np.matrix(W0))
            mean_x = self._update_meanx(var_y, np.matrix(W0), m, cov_x)
            mean_y = self._update_meany(np.matrix(W0), mean_x) 
            W1 = self._update_W(cov_x, mean_x, mean_y)
            var_y = self._update_vary(W1, cov_x, mean_x, mean_y)
            supnorm = np.max(np.abs(W1-W0))
            print supnorm
            W0 = W1.copy()

