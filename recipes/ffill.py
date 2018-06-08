from numpy import array, arange, where, maximum, isnan, nan

arr = array([4, nan, nan,  2, nan ])
mask = isnan(arr)
idx = where(~mask, arange(len(mask)), 0)
maximum.accumulate(idx, out=idx)
out = arr[idx]

print(out)
# [4. 4. 4. 2. 2.]
