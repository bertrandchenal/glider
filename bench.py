from itertools import chain
from time import time

from numpy import sum
from pandas import DataFrame
from glider import Frame
import psutil


current_process = psutil.Process()

def get_data(factor):
    n = lambda i: 'ham{0} ham{0} foo{0}'.format(i).split()
    # n = lambda i: 'ham ham foo'.format(i).split()
    names = lambda s: list(chain.from_iterable(n(i) for i in range(s)))
    values = [1, 2, 3]
    # Create two sets of arrays
    d1 = {'name': names(100 * factor)}
    d2 = {'name': names(10 * factor)}
    # Add some columns
    new_cols =  [
        (100 * factor, d1, 'abc'),
        (10 * factor, d2, 'fgh')]
    for size, d, names in new_cols:
        for n in names:
            d[n] = values * size
    return d1, d2


def join_bench(factor):
    d1, d2 = get_data(factor)
    print('-- join (%s x %s)--' % (len(d1['a']), len(d2['f'])))

    start = time()
    f1 = Frame(d1)
    f2 = Frame(d2)
    f3 = f1.join(f2, 'name', how='inner')
    glider_time = time() - start


    start = time()
    df1 = DataFrame(d1)
    df2 = DataFrame(d2)
    df3 = df1.merge(df2, on='name')
    pandas_time = time() - start

    print('glider: %.2f / pandas: %.2f' % (glider_time, pandas_time))

    print(len(f3), len(df3))

def groupby_bench(factor):
    print('-- groupby --')
    n = lambda i: 'ham{} ham foo'.format(i % (factor / 1000)).split()
    names = list(chain.from_iterable(n(i) for i in range(factor)))
    data = {
        'name': names,
        'a': [1, 2, 3] * factor,
        'b': [1, 2, 3] * factor,
        'c': [1, 2, 3] * factor,
        'd': [1, 2, 3] * factor,
        'e': [1, 2, 3] * factor,
    }

    start = time()
    f1 = Frame(data)
    f2 = f1.select('name', (sum, 'b'))
    print('glider:', time() - start)

    start = time()
    df1 = DataFrame(data)
    df2 = df1.groupby('name')['b'].sum()
    print('pandas:', time() - start)
    # Output
    # glider: 0.088
    # pandas: 0.264



if __name__ == '__main__':
    for exp in range(1, 5):
        join_bench(10**exp)
    groupby_bench(50000)
