from itertools import chain
from time import time
from pandas import DataFrame
from glider import Frame
import psutil

current_process = psutil.Process()

def get_mem():
    mem = current_process.memory_info()
    return mem.rss

def join_bench():
    print('-- join --')
    arr = [1, 2, 3]
    # Create two sets of arrays
    d1 = {
        'name': ['ham', 'ham', 'foo'] * 1000,
    }
    d2 = {
        'name': ['ham', 'ham', 'spam'] * 100,
    }
    # Add some columns
    for factor, d in [(1000, d1), (100, d2)]:
        for col in 'abcde':
            d[col] = arr * factor

    start = time()
    f1 = Frame(d1)
    f2 = Frame(d2)
    mem = get_mem()
    f3 = f1.join(f2, 'name', how='inner')
    assert len(f3) == 400000
    print('glider:', time() - start, get_mem() - mem)
    del f1
    del f2
    del f3

    start = time()
    f1 = DataFrame(d1)
    f2 = DataFrame(d2)
    mem = get_mem()
    f3 = f1.merge(f2, on='name')
    assert len(f3) == 400000
    print('pandas:', time() - start, get_mem() - mem)
    # Output
    # 0.029 20226048
    # 0.059 39178240


def groupby_bench():
    print('-- groupby --')
    n = lambda i: 'ham{0} ham{0} foo{0}'.format(i).split()
    names = list(chain.from_iterable(n(i) for i in range(1000)))
    data = {
        'name': names,
        'a': [1, 2, 3] * 1000,
        'b': [1, 2, 3] * 1000,
        'c': [1, 2, 3] * 1000,
        'd': [1, 2, 3] * 1000,
        'e': [1, 2, 3] * 1000,
    }

    start = time()
    f1 = Frame(data)
    gr = f1.groupby('name')
    s = sum(len(f) for _, f in gr) # operation needed to force evaluation
    print('glider:', time() - start)

    start = time()
    f1 = DataFrame(data)
    gr = f1.groupby('name')
    s = sum(len(f) for _, f in gr)
    print('pandas:', time() - start)

    # Output
    # 0.030
    # 0.171


if __name__ == '__main__':
    join_bench()
    groupby_bench()
