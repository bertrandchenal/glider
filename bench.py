from time import time
from pandas import DataFrame
# from toolz_frame import Frame
from np_frame import Frame
import psutil

current_process = psutil.Process()

def get_mem():
    mem = current_process.memory_info()
    return mem.wset

if __name__ == '__main__':

    mem = get_mem()
    start = time()
    f1 = Frame({
        'name': ['ham', 'ham', 'foo'] * 1000,
        'a': [1, 2, 3] * 1000,
        'b': [1, 2, 3] * 1000,
        'c': [1, 2, 3] * 1000,
        'd': [1, 2, 3] * 1000,
        'e': [1, 2, 3] * 1000,
    })
    f2 = Frame({
        'name': ['ham', 'ham', 'spam'] * 100,
        'a': [1, 2, 3] * 100,
        'b': [1, 2, 3] * 100,
        'c': [1, 2, 3] * 100,
        'd': [1, 2, 3] * 100,
        'e': [1, 2, 3] * 100,
    })


    f3 = f1.join(f2, 'name')
    print time() - start, get_mem() - mem
    del f1
    del f2
    del f3

    mem = get_mem()
    start = time()
    f1 = DataFrame([
        ('ham', 1, 1, 1, 1, 1),
        ('ham', 2, 2, 2, 2, 2),
        ('foo', 3, 3, 3, 3, 3),
    ]*1000, columns=['name', 'a', 'b', 'c', 'd', 'e'])
    f2 = DataFrame([
        ('ham', 10, 10, 10, 10, 10),
        ('ham', 20, 20, 20, 20, 20),
        ('spam', 30, 30, 30, 30, 30),
    ]*100, columns=['name', 'a', 'b', 'c', 'd', 'e'])

    f3 = f1.merge(f2, on='name')
    print time() - start, get_mem() - mem


# Output
# 0.0299999713898 20226048
# 0.0599999427795 39178240
