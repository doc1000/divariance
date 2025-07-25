import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


def divariance(arr_a, arr_b):
    """dvr: divariance    variance of swapped members of two distributions
    I subtract one from denomanator to match covariance... only meaninful for small sample sizes"""
    a_mean, b_mean = arr_a.sum(axis=-1)/(arr_a.shape[-1]-1), arr_b.sum(axis=-1)/(arr_b.shape[-1]-1)#arr_a.mean(axis=-1), arr_b.mean(axis=-1)
    a_sqr_mean, b_sqr_mean = np.power(arr_a,2).sum(axis=-1)/(arr_a.shape[-1]-1) , np.power(arr_b,2).sum(axis=-1)/(arr_b.shape[-1]-1)#np.power(arr_a,2).mean(axis=-1) , np.power(arr_b,2).mean(axis=-1) 
    dvr = (a_sqr_mean+b_sqr_mean)/2 - a_mean*b_mean
    return dvr

def sdd(arr_a,arr_b):
    """sdd: standard dideviation - root divariance, compare to standard deviation """
    return np.sqrt(divariance(arr_a, arr_b))

def dirrelation(arr_a,arr_b):
    """dirrelation: sigma product/divariance     
    compare to correlation, but measures deviation of magnitude vs direction.
    sigma product is the pearson denominator (sigma_a*sigma_b)
    the inverse (1/d) is truly scaled to correlation, but is scaled [1,inf]"""
    return (arr_a.std(axis=-1)*arr_b.std(axis=-1))/ divariance(arr_a, arr_b)

def dvr_gain(arr_a,arr_b):
    """gain: square root of (inverse dirrelation - 1)   
    "divergence";  normalized measure of difference. 
    To be truly consistent with other metrics, dirrelation needs to be inverted so product of stdevs is
    in denomenator (similar to correlation).  The square root is taken to put it in 'standard deviation'
    terms instead of the squared space of variance.
    
    double this and it approximates KL divergence between two somewhat normal distributions"""
    return (np.sqrt(1/dirrelation(arr_a,arr_b))-1)

def _vnorm(arr_a):
    return np.sqrt((np.square(arr_a).sum(axis=-1)))

def _vnorm_prep(v0,v1):
    sm0 = _vnorm(v0) 
    sm1 = _vnorm(v1) 
    if len(v0.shape) < 2:
        sm0 = (np.array([sm0]))
    if len(v1.shape) < 2:
        sm1 = (np.array([sm1]))
    return sm0,sm1
    
def msm(v0,v1):
    """mean squared magnitude: (|X|^2+|Y|^2)/2
    average squared magnitude of two (or more) vectors;  core operator for divariance;  compare to dot product """
    sm0,sm1 = np.square(_vnorm_prep(v0,v1))
    
    return (sm0[:,np.newaxis]+sm1[np.newaxis,:])/2

def sigma_product(v0,v1):
    """product of standard deviations of two distributions.  Denomenator of Pearson Correlation"""
    sm0,sm1 = _vnorm_prep(v0,v1)
      
    return sm0[:,np.newaxis]*sm1[np.newaxis,:]

def magnitude_similarity(v0,v1):
    """Measure of similarity of two vectors by magnitude, independent of direction
    std(v0)*std(v1) / mean squared magnitude
    """
    return sigma_product(v0,v1)/msm(v0,v1)

def vector_similarity(v0,v1):
    """normalized measure of how two vectors are similar in both direction and magnitude 
    cosine_similarity * magnitude similarity
    """
    return cosine_similarity(v0,v1) * magnitude_similarity(v0,v1)

def fid(act1, act2):
	# calculate mean and covariance statistics
    """frechet inception distance numpy implementation
    adapted from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ 
    I changed the rowvar to =True so that the covariance matrix is captured over the features instead of the samples.  
    Since FID is cumulative, you're getting a massive number. That is capturing the covariance between each of the samples,
    not each of the features (activations).Since we're trying to measure the relative relationship between each feature within 
    the collection (and then compare to the other collection), you have to do it feature-wise. 
    In that article cited I think they talk about 'walking the dog'... determining how well the leash length between 
    the first distribution (the walker) and the second (the dog) stays constant."""
    
    mu1, sigma1 = act1.mean(axis=1), cov(act1, rowvar=True)
    mu2, sigma2 = act2.mean(axis=1), cov(act2, rowvar=True)
	# calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
    	covmean = covmean.real
	# calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def vardist(act1,act2,return_features=False):
    """variational inception distance - modeled on frechet inception distance using divariance and shared covariance between two distributions of multi-feature data
    should have application whenever there are multiple samples over multiple features and two distinct categories... 
    that applies to tuning models on the same set of image/text data, classifying based on multiple features of the same model, clustering observations within a single data set/model.
    divariance(act1,act2).sum() - dot_covariance.sum()
    if return_features==True, return the difference by feature, otherwise return the sum of all features
    As the feature means, variances and correlations between the two distributions converge, this will converge to 0
    """
    if return_features:
        return divariance(act1,act2) - frechet_covariance(act1,act2)
    else:
        return divariance(act1,act2).sum() - frechet_covariance(act1,act2).sum()

def frechet_covariance(act1,act2):
    """feature wise shared covariance between two multi-dimensional distributions. Captures the degree to which there is covariance within each distribition 
    and how the covariance compares between the distributions.
    diagnoal of sqrt(Covar_matrix_a .dot covar_matrix_b) 
    find out what the paper called this term.  I could call it dotco for short.  dot implies two linked distributions... we'll try it.
    no reason to calculate the covmean intermediate step.  The trace will be a shortcut to the sum... could integrate it, but not necessary now I think
    As the feature correlation pattern between the two models converges, this value will converge to the product of the variances of the distributions... 
    or the product of the diagonals of the root internal covariance matrices
    """
    sigma1, sigma2 = cov(act1, rowvar=True), cov(act2, rowvar=True)
    return np.diag(sqrtm(sigma1.dot(sigma2)).real)

def frechet_dicorrelation(act1,act2):
    return frechet_covariance(act1,act2).sum()/divariance(act1,act2).sum()
