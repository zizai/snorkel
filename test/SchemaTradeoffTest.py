import numpy as np
import timeit
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', 
                        '.*', 
                        UserWarning,
                        'warnings_filtering',
                        )

candidate_count = 10
feat_count = 10000

iterations = [1, 3, 10, 30, 100]
inv_densities = [1, 10, 100, 1000]
scale = [100, 1000, 10000]
default_inv_density = 100

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

def show_plot(xname):
    axes = plt.gca()
    caption = '%dx%d, density=%s' % (candidate_count, feat_count, str(1.0/default_inv_density))
    axes.set_title(caption)
    axes.set_xlabel(xname.title() +'(log scale)')
    axes.set_ylabel('Execution time(log scale)')
    plt.xscale('log', nonposy='clip')
    plt.yscale('log', nonposy='clip')
    plt.legend(loc='upper left')
    plt.show()
    
def run_benchmark(benchmark):    
    series = {}
    mat_types = [dense, coo, lil]
    
    if benchmark == 'init':
        xs = scale
    elif benchmark == 'iterations':
        xs = iterations
    elif benchmark == 'sparsity':
        xs = inv_densities
        
    for mat in mat_types:
        mat_name = mat.__name__
        series[mat_name] = []
        # x is the variable on x-axis for plotting
        for x in xs:
            repetition = 10
            exeuctions = 1
            if benchmark == 'init':
                run = lambda: mat(x, feat_count)
            elif benchmark == 'iterations':
                def run():
                    # Here we simulate higher dimension with dense matrix
                    m = mat(candidate_count, feat_count)
                    for _ in xrange(x):
                        m.add_col(1)
            elif benchmark == 'sparsity':
                repetition = 100
                exeuctions = 100
                m = mat(candidate_count, feat_count, x)
                run = lambda: m.query()
            timings = timeit.Timer(run).repeat(repetition, exeuctions)
            # min timings are indicative of the hardware limitations
            series[mat_name].append(min(timings))
        plt.plot(xs, series[mat_name], label=mat_name)
    # For plotting purposes we use actual sparsity
    if xs is inv_densities:
        xs = [1.0/x for x in xs]
    print 'Benchmark_' + benchmark + '\t' + '\t'.join([str(x) for x in xs])
    for mat_name, data in series.iteritems():
        print mat_name + '\t' + '\t'.join([str(x) for x in data])
    show_plot(benchmark)
#     plt.clf()
#                 print '\t'.join([mat, '%dx%d' % (size, size), str(iteration), str(avg) + 's'])
#                 print 'mat type', mat
#                 print 'size %dx%d' % (size, size)
#                 print 'col add iterations', iteration
#                 print 'avg time', avg

#     plt.show()

if __name__ == '__main__':
    run_benchmark('iterations')
    run_benchmark('sparsity')
    
    
