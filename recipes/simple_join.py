# Shows how to do a faster join if the join is done on one (and only
# one) numerical column

from time import time
from numpy import unique,  concatenate
from numpy import array, meshgrid
from random import choice

SIZE = 3

values = range(SIZE * SIZE)

ar_left = array([float(choice(values)) for _ in range(SIZE)])
ar_right = array([float(choice(values)) for _ in range(SIZE * SIZE)])

# General join
s = time()
ar_all = concatenate([ar_left, ar_right])
bins, idx = unique(ar_all, return_inverse=True)
idx_l = idx[:len(ar_left)]
idx_r = idx[len(ar_left):]
mg_l, mg_r = meshgrid(idx_l, idx_r, sparse=True)
mg_mask = mg_l == mg_r
keep_r, keep_l = mg_mask.nonzero()
import pdb;pdb.set_trace()

print(time() - s)
print(all(ar_left[keep_l] == ar_right[keep_r]))

# Join on num col
s = time()
mg_l, mg_r = meshgrid(ar_left, ar_right, sparse=True)
mg_mask = mg_l == mg_r
keep_r, keep_l = mg_mask.nonzero()
print(time() - s)
print(all(ar_left[keep_l] == ar_right[keep_r]))
