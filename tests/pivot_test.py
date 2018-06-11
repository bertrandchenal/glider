from pytest import raises

from numpy import isin, nan, sum
from glider import Frame


f1 = Frame({
    'x': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y' : [10, 10, 10, 20, 20, 20, 30, 30, 30],
    'z': [11, 12, 13, 21, 22, 23, 31, 32, 33],
})


def test_simple_pivot():
    f2 = f1.pivot('z', by='y')
    expected = Frame({
        'x': [1, 2, 3],
        10 : [11, 12, 13],
        20: [21, 22, 23],
        30: [31, 32, 33],
    })

    assert f2.equal(expected)

def test_sparse_pivot():
    diag_f = f1.mask(isin(f1['z'], (11, 22, 33)))
    f2 = diag_f.pivot('z', by='y').sorted('x')
    expected = Frame({
        'x': [1, 2, 3],
        10: [11, nan, nan],
        20: [nan, 22, nan],
        30: [nan, nan, 33],
    })
    assert f2.equal(expected)

    triang_f = f1.mask(~isin(f1['z'], (11, 22, 33)))
    f2 = triang_f.pivot('z', by='y')
    expected = Frame({
        'x': [1, 2, 3],
        10: [nan, 12, 13],
        20: [21, nan, 23],
        30: [31, 32, nan],
    })
    assert f2.equal(expected)


def test_duplicated_pivot():
    # Create frame with duplicate values
    f1.append(f1)

    # Make sure we raise an exception if no aggregate is given
    with raises(ValueError):
        f1.pivot('z', by='y')

    # Check aggregation
    f2 = f1.pivot('z', by='y', agg=sum)
    expected = Frame({
        'x': [1, 2, 3,],
        10: [22, 24, 26],
        20: [42, 44, 46],
        30: [62, 64, 66],
    })
    assert f2.equal(expected)
