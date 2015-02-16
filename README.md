# pca-magic
Often, you want to use PCA but your lovely matrix is smattered with NaNs everywhere.

If you don't have too many NaNs, you could try filling in the NaNs with means or some other interpolated value but if you have too many NaNs, your rudimentary interpolation is going to overwhelm the signal in the data with noise.  (Think about the limiting case with all but one NaN).

A better way: suppose you had the latent factors representing the matrix. Construct a linear model for each series and then use the resulting model for interpolation.  Intuitively, this will preserve the signal from the data as the interpolated values come from latent factors. 

However, the problem is you never have these factors to begin with.  The old chicken and egg problem.  But no matter, fixed point algorithm to the rescue.

Install via pip:
```
pip install ppca
```

