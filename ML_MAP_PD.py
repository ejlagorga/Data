#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Implementation of ln with 0 in domain
def ln(x):
  if x == 0:
    return -math.inf
  else:
    return math.log(x)

# Maximum-likelihood estimate
def ml(k, m, a, N, K):
  return m[k]/N

# Maximum a posterior estimation
def mp(k, m, a, N, K):
  return (m[k] + a[k] - 1)/(N + a[0] - K)

# Predictive distribution estimation
def pd(k, m, a, N, K):
  return (m[k] + a[k])/(N + a[0])

# Gamma function
def r(n):
  if n == 0:
    return math.inf
  else:
    return math.factorial(n - 1) 

# Training data class
class TrainingData:

  # Initialize vocabulary
  def __init__(self, vocabulary, training, **kwargs):
    a = kwargs.get('a', 2)
    hi_freq = kwargs.get('hi_freq', None)
    
    # Construct dictionary with words as key and # of
    # occurrences in training data as value
    words = list(set(vocabulary))
    self.vocab = dict(zip(words, [0]*len(words)))
    for word in training:
      self.vocab[word] += 1 

    if hi_freq:
      for word in training:
        if self.vocab[word] < 50:
          self.vocab[word] = 0
    
    # Construct dictionary with words as key and 
    # fixed prior (2) as value
    self.a = dict(zip([0] + words, [a*len(words)] + [a]*len(words)))

  def perplexity(self, model, data):
    m = self.vocab
    a = self.a
    N = len(data)
    K = len(self.vocab)
    sum_lnpr = sum(map(lambda k:ln(model(k, m, a, N, K)), data))
    return math.exp((-1 / N) * sum_lnpr)

  def ln_evidence(self, data):
    m = self.vocab
    a = self.a
    N = len(data)
    K = len(self.vocab)
    ln_prod_r_am = sum(map(lambda k:ln(r(a[k] + m[k])), data))
    ln_prod_r_a  = sum(map(lambda k:ln(r(a[k])), data))
    return (ln(r(a[0])) + ln_prod_r_am) - (ln(r(a[0] + N)) + ln_prod_r_a)


# Read in data files and parse into lists
train = open("data/training_data.txt", 'r').read().split()
test = open("data/test_data.txt", 'r').read().split()
vocab = train + test
              
# Initialize t#raining data for all partitions
trainingData_n    = TrainingData(vocab, train)
trainingData_n4   = TrainingData(vocab, train[:int(len(train)/4)])
trainingData_n16  = TrainingData(vocab, train[:int(len(train)/16)])
trainingData_n64  = TrainingData(vocab, train[:int(len(train)/64)])
trainingData_n128 = TrainingData(vocab, train[:int(len(train)/128)])

# Run tests for Maximum-likelihood estimate
ml_results = [
    trainingData_n.perplexity(ml, test),
    trainingData_n4.perplexity(ml, test),
    trainingData_n16.perplexity(ml, test),
    trainingData_n64.perplexity(ml, test),
    trainingData_n128.perplexity(ml, test)]
print(ml_results)
plt.plot([640000, 160000, 40000, 10000, 5000], ml_results, 'go')

# Run tests for Maximum a posterior estimation
mp_results = [
    trainingData_n.perplexity(mp, test),
    trainingData_n4.perplexity(mp, test),
    trainingData_n16.perplexity(mp, test),
    trainingData_n64.perplexity(mp, test),
    trainingData_n128.perplexity(mp, test)]
print(mp_results)
plt.plot([640000, 160000, 40000, 10000, 5000], mp_results, 'b')

# Run tests for Predictive distribution estimation
pd_results = [
    trainingData_n.perplexity(pd, test),
    trainingData_n4.perplexity(pd, test),
    trainingData_n16.perplexity(pd, test),
    trainingData_n64.perplexity(pd, test),
    trainingData_n128.perplexity(pd, test)]
print(pd_results)
plt.plot([640000, 160000, 40000, 10000, 5000], pd_results, 'r')

# Generate graph to plot results
green = mpatches.Patch(color='green', label='maximum likelihood')
blue = mpatches.Patch(color='blue', label='maximum posterior')
red = mpatches.Patch(color='red', label='predictive distribution')
plt.legend(handles=[green, blue, red])

plt.ylabel('perplexity calculation')
plt.xlabel('size of data set used for training')
plt.show()

a_vals  = []
ev_vals = []
p_vals  = []
for a in list(range(1,11)):
  trainingData_n128 = TrainingData(vocab, train[:int(len(train)/128)], a=a)
  a_vals.append(a)
  ev_vals.append(trainingData_n128.ln_evidence(test))
  p_vals.append(trainingData_n128.perplexity(pd, test))
print(zip(a_vals, ev_vals, p_vals))

# Generate graph to plot results
plt.plot(a_vals, ev_vals, 'r')
plt.title('log evidence as a function of alpha')
plt.xlabel('alpha')
plt.show()

plt.plot(a_vals, p_vals, 'b')
plt.title('predictive distribution perplexity as a func of alpha')
plt.xlabel('alpha')
plt.show()
  
# Authon Prediction 
pg121  = open("data/pg121.txt.clean", 'r').read().split()
pg1400 = open("data/pg1400.txt.clean", 'r').read().split()
pg141  = open("data/pg141.txt.clean", 'r').read().split()
vocab = pg121 + pg1400 + pg141

trainingData_n = TrainingData(vocab, pg121)
print(trainingData_n.perplexity(pd, pg1400))

trainingData_n = TrainingData(vocab, pg121)
print(trainingData_n.perplexity(pd, pg141))

trainingData_n = TrainingData(vocab, pg121, hi_freq='True')
print(trainingData_n.perplexity(pd, pg1400))

trainingData_n = TrainingData(vocab, pg121, hi_freq='True')
print(trainingData_n.perplexity(pd, pg141))




