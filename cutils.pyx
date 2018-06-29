cimport cython
from numpy import empty, concatenate, int64
from collections import deque



@cython.boundscheck(False)
@cython.wraparound(False)
cdef new_chunk(Py_ssize_t chunk_size):
    cdef Py_ssize_t [:,:] chunk
    chunk = empty(shape=(chunk_size, 2), dtype=int64)
    return chunk

@cython.boundscheck(False)
@cython.wraparound(False)
def joiner(Py_ssize_t [:] left, Py_ssize_t [:] right, int chunk_size=1000):
    dq = deque()

    cdef Py_ssize_t [:,:] chunk
    cdef Py_ssize_t idx_l = 0
    cdef Py_ssize_t idx_r = 0
    cdef Py_ssize_t chunk_idx = 0
    cdef Py_ssize_t len_l = left.shape[0]
    cdef Py_ssize_t len_r = right.shape[0]

    chunk = new_chunk(chunk_size)
    dq.append(chunk)

    while idx_l < len_l and idx_r < len_r :
        if left[idx_l] == right[idx_r]:
            # If values match, we add positions to result
            chunk[chunk_idx, 0] = idx_l
            chunk[chunk_idx, 1] = idx_r
            chunk_idx += 1
            if chunk_idx == chunk_size:
                chunk = new_chunk(chunk_size)
                dq.append(chunk)
                chunk_idx = 0

        if left[idx_l] > right[idx_r]:
            # Right is lagging
            idx_r += 1
        elif left[idx_l] < right[idx_r]:
            # Left is lagging
            idx_l += 1
        elif idx_l + 1 < len_l and left[idx_l] == left[idx_l + 1]:
            # Left is on a plateau
            idx_l += 1
        else:
            idx_r += 1

    # truncate last chunk
    dq.append(dq.pop()[:chunk_idx])
    return concatenate(dq)
