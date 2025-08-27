import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.special import kl_div


class Formulas:
    """Calculating divariance based divergence statistics from 1st and 2nd moments of two distributions.
    If passing arrays, use the Divariance.Arrays sister class
    
    Divariance is the combined variance of two distributions if the variance of each observations is calculated
    relative to the membership of the other distributions, rather than the mean.  The squared differences of each (n) observerations
    in X vs each (m) observations in Y are accumulated into a rectangular matrix via double summation over all (n) and (m).
    
    divariance(var-sigma) = [Summation_x=i^n Summation_y=j^m (x_i - y_j)**2) ]/(2mn)

    This can be simplified to a function of moments:

    divariance = [(mean1-mean2)**2 + var1 + var2]/2
    
    Most metric methods takes parameters:
    -------------------------------------
    mean1: float
        mean of first distribution
    mean2: float
        mean of second distribuition
    var1: float
        variance of first distribution
    var2: float
        variance of second distribuition

    Methods
    -------
    divariance(mean1,mean2,var1, var2)
        returns divariance, a metric of divergence between two distributions scaled the same as variance
    dirrelation(mean1,mean2,var1, var2)
        returns dirrelation, divariance scaled by the product of sigma1*sigma2, similarly to correlation
    frechet_distance_norms(mean1,mean2,var1, var2)
        returns frechet distance applied to normal-like distributions
    frechet_divergence(mean1,mean2,var1, var2)
        returns frechet_distance_norms scaled by the product of sigmas
    RDD(mean1,mean2,var1, var2):
        returns root dirrelation divergence, a metric similar to frechet_divergence but with dampened gradients
    divergence_dampening(x,y=2)
    """
    def __init__(self):
        #attributes
        pass
    #methods
    @staticmethod
    def divariance(mean1,mean2,var1, var2,**_):
        '''returns divariance, a metric of divergence between two distributions scaled the same as variance'''
        return (np.power(mean1-mean2,2)+var1+var2)/2
    
    @classmethod
    def dirrelation(cls,mean1,mean2,var1, var2,**_):
        '''returns dirrelation, divariance scaled by the product of sigma1*sigma2, similarly to correlation'''
        return cls.divariance(mean1,mean2,var1, var2)/(np.sqrt(var1*var2))
    
    @staticmethod
    def frechet_distance_norms(mean1,mean2,var1, var2,**_):
        '''returns frechet distance applied to normal-like distributions'''
        return (np.power(mean1-mean2,2)+np.power(np.sqrt(var1)-np.sqrt(var2),2))

    @classmethod
    def frechet_divergence(cls,mean1,mean2,var1, var2,**_):
        '''returns frechet_distance_norms scaled by the product of sigmas, can be calculated as (dirrelation -1)'''
        return cls.frechet_distance_norms(mean1,mean2,var1, var2) /    (2*np.sqrt(var1*var2))
    
    @classmethod
    def root_dirrelation_divergence(cls,mean1,mean2,var1, var2,**_):
        '''returns root dirrelation divergence (rdd), a metric similar to frechet_divergence but with dampened gradients
        can be considered a form of gain based on root dirrelation'''
        return cls.divergence_dampening(value=cls.dirrelation(mean1,mean2,var1, var2), scale=2)

    @staticmethod
    def divergence_dampening(value,scale=2):
        """ applies a dampening functions to divergence to mitigate exploding gradients which diverge from empirical values
            of mean KL divergence due to very small sigmas in denomenators
            value: divergence value with minimum of 1 (divergence+1)
            scale: dampening metric.  Integers will converge to a fixed value quickly so 2 is recommended.  
                Pass y=(x-1) if dynamic dampening is desired, but result may not be convex, seems to match empirical KL divergence relatively well
        """
        scale = max(scale,0.002)
        return np.round(scale * (np.power(value,1/(scale))-1),3)
    
    @classmethod
    def kl_normal(cls,mean1,mean2,var1, var2,return_mean=False,mean_type='arithmetic',**_):
        '''returns the KL Divergence of two normal distributions with the indicated statistics in the direction (KL dist1|dist2).
        KL divergence in directional; this assumes that distribution 1 is the base distribution from which the distance is calculated.
        The formulaic version tends to diverge substantially from empirical KL divergence as the difference in variance gets extreme

        if return_mean = True, then the mean of the KL Divergence from both directions (P|Q & Q|P) will be calcualted.  The arithmatic mean is the default.
                
        Source: Frechet Applied to Multi-variate normal distributions.   Dowson & Landau. 1982.  JOURNAL OF MULTIVARIATE ANALYSIS 12, 450-455 (1982)
        '''
        if not return_mean:
            return 0.5*((var1+np.power(mean1-mean2,2))/var2 - np.log(var1/var2) - 1)
        else:
            if mean_type=='arithmetic':
                return 0.5*( (var1+np.power(mean1-mean2,2))/(2*var2) + (var2+np.power(mean1-mean2,2))/(2*var1) -1)
            if mean_type=='geometric':
                return np.sqrt(cls.kl_normal(mean1,mean2,var1, var2)*cls.kl_normal(mean2,mean1,var2, var1))

class Empirical:
    """Calculating divariance based and other divergence statistics from arrays/samples of two distributions.
    If passing statistics (mean,variance) to formula, use the Divariance.Formulas sister class
    
    Divariance is the combined variance of two distributions if the variance of each observations is calculated
    relative to the membership of the other distributions, rather than the mean.  The squared differences of each (n) observerations
    in X vs each (m) observations in Y are accumulated into a rectangular matrix via double summation over all (n) and (m).
    
    divariance(var-sigma) = [Summation_x=i^n Summation_y=j^m (x_i - y_j)**2) ]/(2mn)

    This can be simplified to a function of moments:

    divariance = [(mean1-mean2)**2 + var1 + var2]/2
    
    Most metric methods take parameters:
    -------------------------------------
    arr_a: numpy array
    arr_b: numpy array

    Methods
    -------
    divariance(mean1,mean2,var1, var2)
        returns divariance, a metric of divergence between two distributions scaled the same as variance
    dirrelation(mean1,mean2,var1, var2)
        returns dirrelation, divariance scaled by the product of sigma1*sigma2, similarly to correlation
    frechet_distance_norms(mean1,mean2,var1, var2)
        returns frechet distance applied to normal-like distributions
    frechet_divergence(mean1,mean2,var1, var2)
        returns frechet_distance_norms scaled by the product of sigmas
    RDD(mean1,mean2,var1, var2):
        returns root dirrelation divergence, a metric similar to frechet_divergence but with dampened gradients
    divergence_dampening(x,y=2)

    kl_divergence(cls,x,y,bins=100,calculator='scipi'):
        returns KL Divergence of two 1-d arrays of raw values (not distributions)
    """
    def __init__(self):
        #attributes
        pass
    #methods
    @staticmethod
    def divariance(arr_a, arr_b):
        """dvr: divariance    variance of swapped members of two distributions
        I subtract one from denomanator to match covariance... only meaninful for small sample sizes"""
        a_mean, b_mean = arr_a.sum(axis=-1)/(arr_a.shape[-1]-1), arr_b.sum(axis=-1)/(arr_b.shape[-1]-1)#arr_a.mean(axis=-1), arr_b.mean(axis=-1)
        a_sqr_mean, b_sqr_mean = np.power(arr_a,2).sum(axis=-1)/(arr_a.shape[-1]-1) , np.power(arr_b,2).sum(axis=-1)/(arr_b.shape[-1]-1)#np.power(arr_a,2).mean(axis=-1) , np.power(arr_b,2).mean(axis=-1) 
        dvr = (a_sqr_mean+b_sqr_mean)/2 - a_mean*b_mean
        return dvr
    
    @classmethod
    def sdd(cls,arr_a,arr_b):
        """sdd: standard dideviation - root divariance, compare to standard deviation """
        return np.sqrt(cls.divariance(arr_a, arr_b))
    
    @classmethod
    def dirrelation(cls,arr_a,arr_b):
        """dirrelation: sigma product/divariance     
        compare to correlation, but measures deviation of magnitude vs direction.
        sigma product is the pearson denominator (sigma_a*sigma_b)
        the inverse (1/d) is truly scaled to correlation, but is scaled [1,inf]"""
        #the following is simpler to understand, but calcuting components once is much faster (3x)
        #return cls.divariance(arr_a, arr_b)/(arr_a.std(axis=-1)*arr_b.std(axis=-1)) 
        a_mean, b_mean = arr_a.sum(axis=-1)/(arr_a.shape[-1]-1), arr_b.sum(axis=-1)/(arr_b.shape[-1]-1)#a_mean, b_mean = arr_a.mean(axis=-1), arr_b.mean(axis=-1)
        a_sqr_mean, b_sqr_mean = np.power(arr_a,2).sum(axis=-1)/(arr_a.shape[-1]-1) , np.power(arr_b,2).sum(axis=-1)/(arr_b.shape[-1]-1)#np.power(arr_a,2).mean(axis=-1) , np.power(arr_b,2).mean(axis=-1) 
        dvr = (a_sqr_mean+b_sqr_mean)/2 - a_mean*b_mean
        return dvr/np.sqrt((a_sqr_mean-a_mean**2)*(b_sqr_mean-b_mean**2))

    @classmethod
    def root_dirrelation_divergence(cls,arr_a,arr_b):
        '''returns root dirrelation divergence (rdd), a metric similar to frechet_divergence but with dampened gradients
        can be considered a form of gain based on root dirrelation'''
        return Formulas.divergence_dampening(value=cls.dirrelation(arr_a,arr_b), scale=2)
    
    @staticmethod
    def _kl(a, b):
        '''simple implementation of KL divergence based on formula
        a: np.array (bin, probability)
            probability density distribution
        b: np.array (bin, probability)
            probability density distribution
        a and b should share the same bins
        '''
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a_mask = np.where(a>0,True,False)
        b_mask = np.where(b>0,True,False)
        return np.sum(np.where(a_mask*b_mask,a * np.log(a / b), 0))
    @classmethod
    def kl_divergence(cls,arr_a,arr_b,bins=100,calculator='scipi'):
        '''Calculated empirical KL divergence from two numpy arrays.  A probability density is calculated
        for each array based on bins determined from min and max values of the arrays.
        arr_a: 1-d numpy array
        arr_b: 1-d numpy array
        bins: int
            number of bins to divide array over (can't exceed length of shortest array)
        calculator: 'scipi' or 'code'
            determines how KL Divergence is calculated.  should be equivalent
        '''
        min_val = min(np.min(arr_a),np.min(arr_b))-0.01
        max_val = max(np.max(arr_a),np.max(arr_b))
        x_pd, _ = np.histogram(arr_a, density=True,bins=bins,range=(min_val,max_val))
        y_pd, _ = np.histogram(arr_b, density=True,bins=bins,range=(min_val,max_val))

        if calculator == 'scipi':
            return np.sum(np.nan_to_num(kl_div(x_pd,y_pd),posinf=0))
        if calculator == 'code':
            return cls._kl(x_pd,y_pd)
    

class Vectors:
    def __init__():
        pass
    def _vnorm(arr_a):
        return np.sqrt((np.square(arr_a).sum(axis=-1)))
    @classmethod
    def _vnorm_prep(cls,v0,v1):
        sm0 = cls._vnorm(v0) 
        sm1 = cls._vnorm(v1) 
        if len(v0.shape) < 2:
            sm0 = (np.array([sm0]))
        if len(v1.shape) < 2:
            sm1 = (np.array([sm1]))
        return sm0,sm1
    @classmethod
    def msm(cls,v0,v1):
        """mean squared magnitude: (|X|^2+|Y|^2)/2
        average squared magnitude of two (or more) vectors;  core operator for divariance;  compare to dot product """
        sm0,sm1 = cls._vnorm_prep(v0,v1)
        sm0,sm1 = np.square(sm0),np.square(sm1)
        
        return (sm0[:,np.newaxis]+sm1[np.newaxis,:])/2
    @classmethod
    def sigma_product(cls,v0,v1):
        """product of standard deviations of two distributions.  Denomenator of Pearson Correlation"""
        sm0,sm1 = cls._vnorm_prep(v0,v1)
        
        return sm0[:,np.newaxis]*sm1[np.newaxis,:]
    @classmethod
    def magnitude_similarity(cls,v0,v1):
        """Measure of similarity of two vectors by magnitude, independent of direction
        std(v0)*std(v1) / mean squared magnitude
        """
        return cls.sigma_product(v0,v1)/cls.msm(v0,v1)
    @classmethod
    def vector_similarity(cls,v0,v1):
        """normalized measure of how two vectors are similar in both direction and magnitude 
        cosine_similarity * magnitude similarity
        """
        return cosine_similarity(v0,v1) * cls.magnitude_similarity(v0,v1)

class Frechet:
    @staticmethod
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

    @classmethod
    def vardist(cls,act1,act2,return_features=False):
        """variational inception distance - modeled on frechet inception distance using divariance and shared covariance between two distributions of multi-feature data
        should have application whenever there are multiple samples over multiple features and two distinct categories... 
        that applies to tuning models on the same set of image/text data, classifying based on multiple features of the same model, clustering observations within a single data set/model.
        divariance(act1,act2).sum() - dot_covariance.sum()
        if return_features==True, return the difference by feature, otherwise return the sum of all features
        As the feature means, variances and correlations between the two distributions converge, this will converge to 0
        """
        if return_features:
            return Empirical.divariance(act1,act2) - cls.frechet_covariance(act1,act2)
        else:
            return Empirical.divariance(act1,act2).sum() - cls.frechet_covariance(act1,act2).sum()
    
    @staticmethod
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

    @classmethod
    def frechet_dicorrelation(cls,act1,act2):
        return cls.frechet_covariance(act1,act2).sum()/Empirical.divariance(act1,act2).sum()
