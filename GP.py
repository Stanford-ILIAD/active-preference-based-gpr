from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import norm
from util import *


class GaussianProcess:
    def __init__(self, initialPoint=0, theta=0.1, noise_level=0.1):
        self.listQueries =[] #list of queries
        self.K = np.zeros((2,2)) #Covariance matrix for our queries
        self.Kinv = np.zeros((2, 2)) #inverse of that covariance matrix
        self.fqmean = 0 #posterior mean for the queries
        self.theta = theta #hyperparameter
        self.W = np.zeros((2,2)) #hessian at the queries
        self.noise = noise_level
        self.initialPoint = np.array(initialPoint)  #initial point that is set to have a 0 value
        self.dim = len(self.initialPoint) #number of features

    def updateParameters(self,query,answer):
        self.listQueries.append([query[0],query[1],answer])
        self.K = self.covK()
        self.Kinv = inv(self.K+np.identity(2*len(self.listQueries))*1e-8) #adding the 1e-8 for numerical stability
        self.fqmean = self.meanmode()
        self.W = self.hessian()


    def objectiveEntropy(self,x): #Compute the objective function (entropy) for a query [xa,xb]
    #we want to maximize this function for bestQuery
        xa = x[:self.dim]
        xb = x[self.dim:]

        matCov = self.postcov(xa, xb)
        mua, mub = self.postmean(xa, xb)
        sigmap = np.sqrt(np.pi * np.log(2) / 2)*self.noise

        result1 = h(
            phi((mua - mub) / (np.sqrt(2*self.noise**2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))))
        result2 = sigmap * 1 / (np.sqrt(sigmap ** 2 + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1])) * np.exp(
            -0.5 * (mua - mub)**2  / (sigmap ** 2  + matCov[0][0] + matCov[1][1] - 2 * matCov[0][1]))

        return result1 - result2
    
    def kernel(self, xa, xb):
        return 1*(np.exp(-self.theta*np.linalg.norm(np.array(xa) - np.array(xb)) ** 2)) - np.exp(-self.theta*np.linalg.norm(xa-self.initialPoint)**2)*np.exp(-self.theta*np.linalg.norm(xb-self.initialPoint)**2)

    def meanmode(self): #find the posterior means for the queries
        n = len(self.listQueries)
        Kinv = self.Kinv
        listResults = np.array(self.listQueries)[:,2]
        def logposterior(f):
            fodd  = f[1::2]
            feven = f[::2]
            fint = 1/self.noise*(feven-fodd)
            res =np.multiply(fint, listResults)
            res = res.astype(dtype = np.float64)
            res = norm.cdf(res)
            res = np.log(res)
            res = np.sum(res)
            ftransp = f.reshape(-1,1)
            return -1*(res- 0.5 * np.matmul(f, np.matmul(Kinv, ftransp)))


        def gradientlog(f):
            grad = np.zeros(2*len(self.listQueries))
            for i in range(len(self.listQueries)):
                signe = self.listQueries[i][2]
                grad[2*i]= self.listQueries[i][2]*(phip(signe*1/self.noise*(f[2*i]-f[2*i+1]))*1/self.noise)/phi(signe*1/self.noise*(f[2*i]-f[2*i+1]))
                grad[2*i+1] = self.listQueries[i][2]*(-phip(signe*1 / self.noise * (f[2 * i] - f[2 * i + 1])) * 1 / self.noise) / phi(
                    signe*1 / self.noise * (f[2 * i] - f[2 * i + 1]))
            grad = grad - f@Kinv
            return -grad
        x0 = np.zeros(2*n)
        return minimize(logposterior, x0=x0,jac=gradientlog).x


    def hessian(self):
        n = len(self.listQueries)
        W = np.zeros((2*n,2*n))
        for i in range(n):
            dif = self.listQueries[i][2]*1/self.noise*(self.fqmean[2*i]-self.fqmean[2*i+1])
            W[2*i][2*i] = -(1/self.noise**2)*(phipp(dif)*phi(dif)-phip(dif)**2)/(phi(dif)**2)
            W[2*i+1][2*i] = -W[2*i][2*i]
            W[2*i][2*i+1] = -W[2*i][2*i]
            W[2*i+1][2*i+1] = W[2*i][2*i]
        return W


    def kt(self, xa, xb):  #covariance between xa,xb and our queries
        n = len(self.listQueries)
        return np.array([[self.kernel(xa,self.listQueries[i][j])for i in range(n) for j in range(2)], [self.kernel(xb,self.listQueries[i][j])for i in range(n) for j in range(2)]])

    def covK(self): #covariance matrix for all of our queries
        n= len(self.listQueries)
        return np.array([[self.kernel(self.listQueries[i][j], self.listQueries[l][m]) for l in range(n) for m in range(2)] for i in range(n) for j in range(2)])

    def postmean(self, xa, xb): #mean vector for two points xa and xb
        kt = self.kt(xa,xb)
        return np.matmul(kt, np.matmul(self.Kinv,self.fqmean))

    def cov1pt(self,x): #variance for 1 point
        return self.postcov(x,0)[0][0]

    def mean1pt(self,x):
        return self.postmean(x,0)[0]

    def postcov(self, xa,xb): #posterior covariance matrix for two points
        n = len(self.listQueries)
        Kt = np.array([[self.kernel(xa,xa), self.kernel(xa,xb)], [self.kernel(xb,xa), self.kernel(xb,xb)]])
        kt = self.kt(xa,xb)
        W = self.W
        K = self.K
        return Kt - kt@inv(np.identity(2*n)+np.matmul(W,K))@W@np.transpose(kt)
