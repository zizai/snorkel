import numpy as np
from numpy.random import RandomState
import timeit
from scipy.sparse import random
from scipy import stats
import matplotlib.pyplot as plt

class CustomRandomState(object):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

seed = 1337

sizes = [100]  # [10, 100, 1000]
candidate_count = 1000
feat_count = 10
iterations = [1, 3, 10, 30, 100, 300, 1000]
densities = [0.001, 0.01, 0.1, 0.5, 1]

rs = RandomState(seed)
rvs = stats.poisson(25, loc=10).rvs
S = random(3, 4, density=0.25, random_state=rs, data_rvs=rvs)
# Density for 
inv_density = 2

class lol():
    
    @staticmethod
    def new_immutable_vec(size):
        return tuple(range(size))
    
    def __init__(self, n, m):
        self.data = [lol.new_immutable_vec(m) for _ in xrange(n)]
        
    def add_col(self, size):
        for i in xrange(len(self.data)):
            # Updates each row with new cols
            self.data[i] += lol.new_immutable_vec(size)
        
class dok():
    
    def __init__(self, n, m):
        # Simulate (id, key, val) schema
        self.data = [(i, j) for i in xrange(n) for j in xrange(m)]
        self.row_count = n
        
    def add_col(self, size):
        for r in xrange(self.row_count):
            for new_key in xrange(size):
                self.data.append((r, new_key))
                
class dense():
    
    def __init__(self, n, m):
        # Simulate (id, key, val) schema
        self.data = np.random.rand(n, m*inv_density)
        self.n = n
        
    def add_col(self, size):
        effective_size = size * inv_density
        # Effectively copy the old data into a newly allocated matrix
        self.data = np.append(self.data, np.random.rand(self.n, effective_size), axis=1)

def benchmark_add_col(mat_class, iterations, init_size):
    mat = mat_class(candidate_count, feat_count)
    for _ in xrange(iterations):
        mat.add_col(1)

if __name__ == '__main__':
    series = {}
    mat_types = ['lol', 'dok', 'dense']
    for mat in mat_types:
        series[mat] = []
        for size in sizes:
            for iteration in iterations:
                stmt = 'benchmark_add_col(%s,%d,%d)' % (mat, iteration, size)
                setup = 'from __main__ import benchmark_add_col, ' + ', '.join(mat_types)
                avg = np.average(timeit.Timer(stmt, setup=setup).repeat(3, 1))
                series[mat].append(avg)
            plt.plot(iterations, series[mat], label=mat)
#                 print '\t'.join([mat, '%dx%d' % (size, size), str(iteration), str(avg) + 's'])
#                 print 'mat type', mat
#                 print 'size %dx%d' % (size, size)
#                 print 'col add iterations', iteration
#                 print 'avg time', avg
    axes = plt.gca()
    axes.set_title('')
    axes.set_xlabel('Iterations(log scale)')
    axes.set_ylabel('Execution time(log scale)')
#     axes.set_ylim([0, 0.05])
    plt.xscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')
    plt.legend(loc='upper left')
    plt.show()
    print series
