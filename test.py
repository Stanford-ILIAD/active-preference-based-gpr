from GP import GaussianProcess
import numpy as np
import scipy.optimize as opt

def findBestQuery(gp):
    def negative_info_gain(x):
        return -1*gp.objectiveEntropy(x)
    x0 = np.array(gp.initialPoint*2) + np.random.rand(gp.dim*2)
    # Let's now find the optimal query within the bounds (-2,2) for each dimension
    opt_res = opt.fmin_l_bfgs_b(negative_info_gain, x0=x0, bounds=[(-2,2)]*gp.dim*2, approx_grad=True, iprint=-1)
    return opt_res[0], -opt_res[1]

initialPoint = [0,0] # where we assume the function value is 0
theta = 1. # hyperparameter
noise_level = 0.1 # noise parameter that corresponds to \sqrt{2}\sigma in the paper

gp = GaussianProcess(initialPoint, theta, noise_level)
gp.updateParameters([[0,0],[1,0]], 1) # We compare the features [0,0] and [1,0], and the former one is preferred
gp.updateParameters([[2,-1],[3,1]],-1) # We compare the features [2,-1] and [3,1], and the latter one is preferred

print('posterior mean for the feature set [3,1] = ' + str(gp.mean1pt([3,1])))
print('posterior covariance between the features [4,0] and [-1,1] = ' + str(gp.postcov([4,0],[-1,1])))
print('posterior variance of the feature set [-2,1] = ' + str(gp.cov1pt([-2,1])))
print('expected information gain from the query [0,0] vs [2,2] = ' + str(gp.objectiveEntropy([[0,0],[2,2]])))

optimal_query, info_gain = findBestQuery(gp)
print('optimal next query is ' + str(optimal_query))
print('expected information gain from the optimal quer is ' + str(info_gain) + ' bits')



