from numpy import isnan, nan

from glider import Frame


f1 = Frame({
    'x': [2, 3, 4],
    'v': [20, 30, 40],
})

f2 = Frame({
    'x': [1, 2, 3],
    'w': [10, 20, 30],
})

f3 = Frame({
    'x': [1, 2, 2, 3],
    'w': [10, 20, 20, 30],
})


def test_inner_join():
    res = f1.join(f2, 'x', how='inner')
    assert list(res['w']) == [20.0, 30.0]

def test_left_join():
    res = f1.join(f2, 'x')
    assert list(res['w'][:2]) == [20.0, 30.0]
    assert isnan(res['w'][2])

def test_right_join():
    res = f1.join(f2, 'x', how='right')
    assert list(res['v'][:2]) == [20.0, 30.0]
    assert isnan(res['v'][2])

def test_outer_join():
    res = f1.join(f2, 'x', how='outer').sorted('x')
    expected = Frame({
        'x': [1, 2, 3, 4],
        'v': [nan, 20, 30, 40],
        'w': [10, 20, 30, nan],
    })
    assert res.equal(expected)

def test_dup_inner_join():
    res = f1.join(f3, 'x', how='inner')
    assert list(res['w']) == [20.0, 20.0, 30.0]

def test_dup_left_join():
    res = f1.join(f3, 'x')
    assert list(res['w'][:3]) == [20.0, 20.0, 30.0]
    assert isnan(res['w'][3])

def test_dup_right_join():
    res = f1.join(f3, 'x', how='right')
    assert list(res['v'][:3]) == [20.0, 20.0, 30.0]
    assert isnan(res['v'][3])

def test_dup_outer_join():
    res = f1.join(f3, 'x', how='outer').sorted('x')
    expected = Frame({
        'x': [1, 2, 2, 3, 4],
        'v': [nan, 20, 20, 30, 40],
        'w': [10, 20, 20, 30, nan],
    })
    assert res.equal(expected)
