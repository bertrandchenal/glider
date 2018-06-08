from numpy import interp, arange, where, maximum, isnan, nan
from frame import Frame


## Up-sample:

f1 = Frame({
    'x': [1, 3, 7],
    'v': [10, 30, 70],
})
f2 = Frame({'x': range(10)})
f3 = f2.join(f1).sorted('x')
v = interp(f3['x'], f1['x'], f1['v'])
f3['v'] = v

print(f3)
# x -> [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
# v -> [10. 10. 20. 30. 40. 50. 60. 70. 70. 70.]
