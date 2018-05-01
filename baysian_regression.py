import csv
import numpy as np
from numpy import linalg as la
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# Converts csv file into np matrix
def csvToMatrix(fname):
    with open(fname, 'r') as f:
        data = list(csv.reader(f))
    return np.array(data).astype(float)

# Converts data to polynomial of degree
def scalarToPolynomial(data, degree):
    polydata = np.ones(data.shape, dtype=np.float)
    for x in range(1, degree+1):
        polydata = np.hstack([polydata, np.power(data, x)])
    return polydata    

# Takes random sample of size sample from dataset
def sampleMatrices(train, trainR, size):
    data = np.hstack([train, trainR])
    sample = data[np.random.choice(data.shape[0], size=size, replace=False)]
    return np.hsplit(sample, [train.shape[1]])

# Calculate Square Error with data, weights, and target
def squareError(data, dataR, weights):
    sum_sq = 0
    for i in range(0, len(data)):
        sum_sq += (float) (np.matmul(data[i], weights) - dataR[i])**2
    return sum_sq

class TrainingData:

    def __init__(self, train, trainR, test, testR, **kwargs):
        self.train  = csvToMatrix(train)
        self.trainR = csvToMatrix(trainR)
        self.test   = csvToMatrix(test)
        self.testR  = csvToMatrix(testR)

        # If training_set use train data as test data
        if kwargs.get('training_set', False):
            self.test  = self.train  
            self.testR = self.trainR

        # Take sample from training data set, if not passed use full set
        self.train, self.trainR = sampleMatrices(self.train, self.trainR,
                                  kwargs.get('sample_size', self.train.shape[0]))

        # Generate training and test data if polynomial data
        degree = kwargs.get('polynomial', 0)
        if degree:
            self.train = scalarToPolynomial(self.train, degree)
            self.test = scalarToPolynomial(self.test, degree)

        self.II = np.matmul(self.train.T, self.train)
        self.It = np.matmul(self.train.T, self.trainR)

    # Calculate MSE given a form of model selection
    def meanSquareError(self, model, **kwargs):
        if model == 'linear':
            weights = self.pseudoInverse(**kwargs)
        elif model == 'bayes':
            weights = self.bayesianModelSelection(**kwargs)[2]
        else:
            raise Exception('No model selected')
        return squareError(self.test, self.testR, weights)/self.test.shape[1]

    # Calculates log evidence as given by 3.86 Bishop
    def logEvidence(self):
        a,B,mn = self.bayesianModelSelection()
        N,M = self.train.shape
        return ((M/2)*np.log(a) + (N/2)*np.log(B) - np.average(mn) - (N/2)*np.log(2*np.pi)
                   - (1/2)*np.log(la.det(a*np.identity(self.train.shape[1]) + B*self.II)))

    # Calculates pseduoinverse from 3.28 Bishop
    def pseudoInverse(self, **kwargs):
        regCoef = kwargs.get('regular', 0)
        regData = regCoef * np.eye(self.train.shape[1]) + self.II
        return np.matmul(np.linalg.inv(regData), self.It)

    # Calculates mn from Bishop 3.52
    def bayesianModelSelection(self):
        a = 10 * np.random.rand()
        B = 10 * np.random.rand()
        So = la.inv(a*np.identity(self.train.shape[1]) + B*self.II)
        mo = np.matmul(B*So, self.It)
        eigvals = la.eigvals(self.II)
        return self.selectModel(a, B, mo, eigvals)

    # Recursively find mn by waiting for a B to converge
    def selectModel(self, a, B, mn, eigvals):
        print(a, B)
        gamma = sum(map(lambda y: y/(a+y), B*eigvals))
        a1 = gamma / np.asscalar(np.matmul(mn.T, mn))
        B1 = (self.train.shape[0] - gamma)/squareError(self.train, self.trainR, mn)
        Sn = la.inv(a1*np.identity(self.train.shape[1]) + B1*self.II)
        mn = np.matmul(B1*Sn, self.It)
        if (la.norm(np.array([a,B]) - np.array([a1,B1])) < 10**-5):
            return [a1, B1, mn]
        return self.selectModel(a1, B1, mn, eigvals)


# Initialize TraingData
KH = TrainingData('data/train-1000-100.csv', 'data/trainR-1000-100.csv', 
                  'data/test-1000-100.csv', 'data/testR-1000-100.csv')
KHTr = TrainingData('data/train-1000-100.csv', 'data/trainR-1000-100.csv', 
      'data/test-1000-100.csv', 'data/testR-1000-100.csv', training_set=True)

HH = TrainingData('data/train-100-100.csv', 'data/trainR-100-100.csv', 
                  'data/test-100-100.csv', 'data/testR-100-100.csv')
HHTr = TrainingData('data/train-100-100.csv', 'data/trainR-100-100.csv', 
      'data/test-100-100.csv', 'data/testR-100-100.csv', training_set=True)

HT = TrainingData('data/train-100-10.csv', 'data/trainR-100-10.csv', 
                  'data/test-100-10.csv', 'data/testR-100-10.csv')
HTTr = TrainingData('data/train-100-10.csv', 'data/trainR-100-10.csv', 
      'data/test-100-10.csv', 'data/testR-100-10.csv', training_set=True)

crime = TrainingData('data/train-crime.csv', 'data/trainR-crime.csv', 
                     'data/test-crime.csv', 'data/testR-crime.csv')
crimeTr = TrainingData('data/train-crime.csv', 'data/trainR-crime.csv', 
       'data/test-crime.csv', 'data/testR-crime.csv', training_set=True)

wine = TrainingData('data/train-wine.csv', 'data/trainR-wine.csv', 
                    'data/test-wine.csv', 'data/testR-wine.csv')
wineTr = TrainingData('data/train-wine.csv', 'data/trainR-wine.csv', 
      'data/test-wine.csv', 'data/testR-wine.csv', training_set=True)


y = range(0,150)

plt.title('1000-100 Data')
plt.xlabel('Regularization Coefficient')
plt.ylabel('Mean Squared Error')
plot(y, list(map(lambda x: KH.meanSquareError('linear', regular=x), y)), 'r')
plot(y, list(map(lambda x: KHTr.meanSquareError('linear', regular=x), y)), 'b')
plt.legend(handles=[mp.Patch(color='r',label='test'),mp.Patch(color='b',label='train')])
plt.show()

plt.title('100-100 Data')
plt.xlabel('Regularization Coefficient')
plt.ylabel('Mean Squared Error')
plot(y, list(map(lambda x: HH.meanSquareError('linear', regular=x), y)), 'r')
plot(y, list(map(lambda x: HHTr.meanSquareError('linear', regular=x), y)), 'b')
plt.legend(handles=[mp.Patch(color='r',label='test'),mp.Patch(color='b',label='train')])
plt.show()

plt.title('100-10 Data')
plt.xlabel('Regularization Coefficient')
plt.ylabel('Mean Squared Error')
plot(y, list(map(lambda x: HT.meanSquareError('linear', regular=x), y)), 'r')
plot(y, list(map(lambda x: HTTr.meanSquareError('linear', regular=x), y)), 'b')
plt.legend(handles=[mp.Patch(color='r',label='test'),mp.Patch(color='b',label='train')])
plt.show()

plt.title('Crime Data')
plt.xlabel('Regularization Coefficient')
plt.ylabel('Mean Squared Error')
plot(y, list(map(lambda x: crime.meanSquareError('linear', regular=x), y)), 'r')
plot(y, list(map(lambda x: crimeTr.meanSquareError('linear', regular=x), y)), 'b')
plt.legend(handles=[mp.Patch(color='r',label='test'),mp.Patch(color='b',label='train')])
plt.show()

plt.title('Wine Data')
plt.xlabel('Regularization Coefficient')
plt.ylabel('Mean Squared Error')
plot(y, list(map(lambda x: wine.meanSquareError('linear', regular=x), y)), 'r')
plot(y, list(map(lambda x: wineTr.meanSquareError('linear', regular=x), y)), 'b')
plt.legend(handles=[mp.Patch(color='r',label='test'),mp.Patch(color='b',label='train')])
plt.show()


# Find Optimal Regularization Coeffieicent
samples = range(10, 800, 10) 
models = list(map(lambda x:TrainingData('data/train-1000-100.csv','data/trainR-1000-100.csv', 
              'data/test-1000-100.csv', 'data/testR-1000-100.csv', sample_size=x), samples))
y = [1, 27, 80]
mserrors = np.zeros([len(models),len(y)])
for model,i in zip(models, range(0,len(models))):
    mse = np.zeros([1,len(y)])  
    for j in range(0, 20):
        mse += np.array(list(map(lambda x: model.meanSquareError('linear', regular=x), y)))
    mserrors[i] = mse/10

plt.title('Learning Curve 1000-100 Data')
plt.xlabel('Cardinality of Training Set')
plt.ylabel('Mean Squared Error')
plot(samples, mserrors[:,0], 'r') 
plot(samples, mserrors[:,1], 'g') 
plot(samples, mserrors[:,2], 'b') 
plt.legend(handles=[mp.Patch(color='r',label='lambda=2'),
    mp.Patch(color='b',label='lambda=80'), mp.Patch(color='g',label='lambda=27')])
plt.show()


# Compare results from optimal perterbation with the 
# model selection in Baysian Linear Regreassion
print(HH.meanSquareError('linear', regular=3))
print(HH.meanSquareError('bayes'))


polynomials = range(1,11)
f3_models = list(map(lambda x:TrainingData('data/train-f3.csv', 'data/trainR-f3.csv', 
              'data/test-f3.csv', 'data/testR-f3.csv', polynomial=x), polynomials))
f5_models = list(map(lambda x:TrainingData('data/train-f5.csv', 'data/trainR-f5.csv', 
              'data/test-f5.csv', 'data/testR-f5.csv', polynomial=x), polynomials))
data = np.zeros([len(f3_models),3])
for model,i in zip(f3_models, range(0, len(f3_models))):
    data[i,0] = model.logEvidence()
    data[i,1] = model.meanSquareError('bayes')
    data[i,2] = model.meanSquareError('linear') 

plt.title('Model Selection by Maximizing log Evidence on f3 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Log Evidence')
plot(polynomials, data[:,0], 'r') 
plt.show()

plt.title('Bayesian Model Selection on f3 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Mean Squared Error')
plot(polynomials, data[:,1], 'g') 
plt.show()

plt.title('Non-Regularized Model on f3 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Mean Squared Error')
plot(polynomials, data[:,2], 'b') 
plt.show()

f5_models = list(map(lambda x:TrainingData('data/train-f5.csv', 'data/trainR-f5.csv', 
              'data/test-f5.csv', 'data/testR-f5.csv', polynomial=x), polynomials))
data = np.zeros([len(f5_models),3])
for model,i in zip(f5_models, range(0, len(f5_models))):
    data[i,0] = model.logEvidence()
    data[i,1] = model.meanSquareError('bayes')
    data[i,2] = model.meanSquareError('linear')

plt.title('Model Selection by Maximizing log Evidence on f5 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Log Evidence')
plot(polynomials, data[:,0], 'r') 
plt.show()

plt.title('Bayesian Model Selection on f5 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Mean Squared Error')
plot(polynomials, data[:,1], 'g') 
plt.show()

plt.title('Non-Regularized Model on f5 Data')
plt.xlabel('Order of Generated Training Data')
plt.ylabel('Mean Squared Error')
plot(polynomials, data[:,2], 'b') 
plt.show()


