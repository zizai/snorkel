import numpy as np
import timeit
import matplotlib.pyplot as plt
from collections import defaultdict


candidate_count = 10

iterations = [1, 3, 10, 30, 100]
inv_densities = [100, 10, 1]
feat_sizes = [100, 1000, 10000]
default_inv_density = inv_densities[0]

def new_immutable_vec(size):
    return tuple(range(size))

class lil():
    
    def __init__(self, n, m, inv_sparsity = default_inv_density):
        m/=inv_sparsity
        self.data = [new_immutable_vec(m) for _ in xrange(n)]
        
    def add_col(self, size):
        for i in xrange(len(self.data)):
            # Updates each row with new cols
            self.data[i] += new_immutable_vec(size)
            
    def query(self):
        return self.data[0]
        
class coo():
    
    def __init__(self, n, m, inv_sparsity = default_inv_density):
        # Simulate (id, key, val) schema
        m/=inv_sparsity
        self.data = [(i, j) for i in xrange(n) for j in xrange(m)]
        self.row_count = n
        self.col_count = m
        
    def add_col(self, size):
        self.col_count += size
        for r in xrange(self.row_count):
            for new_key in xrange(size):
                self.data.append((r, new_key))
                
    def query(self):
        return list(self.data[:self.col_count])
                
class dense():
    
    def __init__(self, n, m, inv_sparsity = 1):
        # Simulate (id, key, val) schema
        self.data = np.random.rand(n, m)
        self.n = n
        
    def add_col(self, size):
        # Effectively copy the old data into a newly allocated matrix
        self.data = np.append(self.data, np.random.rand(self.n, size), axis=1)
        
    def query(self):
        return self.data[0]

def show_plot(xname, yname):
    axes = plt.gca()
    caption = '%dx%d, density=%s' % (candidate_count, feat_count, str(1.0/default_inv_density))
    axes.set_title(caption)
    axes.set_xlabel(xname.title())
    axes.set_ylabel(yname.title())
    plt.xscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')
    plt.legend(loc='upper left')
    plt.show()
    
def run_benchmark(xaxis, yaxis):    
    series = defaultdict(list)
    mat_types = [dense, coo, lil]
    
    if xaxis == 'feat_size':
        xs = feat_sizes
    elif xaxis == 'iterations':
        xs = iterations
    elif xaxis == 'sparsity':
        xs = inv_densities
        
    for mat in mat_types:
        mat_name = mat.__name__
        # x is the variable on x-axis for plotting
        for x in xs:
            repetition = 10
            exeuctions = 1
            if xaxis == 'feat_size':
                if yaxis == 'mat_time':
                    run = lambda: mat(candidate_count, x)
                else:
                    m = mat(candidate_count, feat_sizes[0])
                    run = lambda: m.query() 
            elif xaxis == 'iterations':
                def run():
                    # Here we simulate higher dimension with dense matrix
                    m = mat(candidate_count, feat_sizes[0])
                    for _ in xrange(x):
                        m.add_col(1)
            elif xaxis == 'sparsity':
                repetition = 100
                exeuctions = 100
                m = mat(candidate_count, feat_sizes[0], x)
                run = lambda: m.query()
                
               
            timings = timeit.Timer(run).repeat(repetition, exeuctions)
            # min timings are indicative of the hardware limitations
            series[mat_name].append(min(timings))
        plt.plot(xs, series[mat_name], label=mat_name)
    
    # Ad-hoc postprocessings    
    # For plotting purposes we use actual sparsity
    if xs is inv_densities: xs = [1.0/x for x in xs]
        
    print 'Benchmark_' + xaxis + '\t' + '\t'.join([str(x) for x in xs])
    for mat_name, data in series.iteritems():
        print mat_name + '\t' + '\t'.join([str(x) for x in data])
    show_plot(xaxis, yaxis)


if __name__ == '__main__':
    xaxis = ['feat_size', 'iterations', 'sparsity']
    yaxis = ['query_time','mat_time']
    for y in yaxis:
        for x in xaxis:
            run_benchmark(x, y)
    
    
