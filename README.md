# pca-magic
Often, you want to use PCA but your lovely matrix is smattered with NaNs everywhere.

If you don't have too many NaNs, you could try filling in the NaNs with means or some other interpolated value but if you have too many NaNs, your rudimentary interpolation is going to overwhelm the signal in the data with noise.  (Think about the limiting case with all but one NaN).

A better way: suppose you had the latent factors representing the matrix. Construct a model for each series/column and then use the resulting model for interpolation.  Intuitively, this will preserve the signal from the data as the interpolated values come from latent factors. 

However, the problem is you never have these factors to begin with.  If you had them, you wouldn't be reading this.  

- form the calculation of the latent factors as a fixed point problem with a (good) initial guess of the factors.  
- iteratively construct the factors. 
