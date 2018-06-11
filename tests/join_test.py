from numpy import isnan

from glider import Frame


f1 = Frame({
    'x': [2, 3, 4],
    'v': [20, 30, 40],
})

f2 = Frame({
    'x': [1, 2, 3],
    'w': [10, 20, 30],
})


def test_left_join():
    f3 = f1.join(f2, 'x')
    assert list(f3['w'][:2]) == [20.0, 30.0]
    assert isnan(f3['w'][2])

def test_right_join():
    f3 = f1.join(f2, 'x', how='right')
    print(f3)
    assert list(f3['v'][:2]) == [20.0, 30.0]
    assert isnan(f3['v'][2])

def test_outer_join():
    f3 = f1.join(f2, 'x', how='outer').sorted('x')
    print()
    print(f3)
    res = '''x -> [1. 2. 3. 4.]
v -> [nan 20. 30. 40.]
w -> [10. 20. 30. nan]'''
    assert str(f3) == res
