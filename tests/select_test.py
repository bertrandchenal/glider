from numpy import min, max
from glider import Frame


f1 = Frame({
    'x': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'y' : [10, 10, 10, 20, 20, 20, 30, 30, 30],
    'z': [11, 12, 13, 21, 22, 23, 31, 32, 33],
})


def test_simple_select():
    f2 = f1.select('x', 'y', 'z')
    assert f2.equal(f1)

    f2 = f1.select('x', 'z')
    expected = Frame({
    'x': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'z': [11, 12, 13, 21, 22, 23, 31, 32, 33],
    })
    assert f2.equal(expected)


def test_aliasing():
    f2 = f1.select(
        ('x', 'ham'),
        ('y', 'spam'),
        ('z', 'foo'),
    )
    expected = Frame({
    'ham': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'spam' : [10, 10, 10, 20, 20, 20, 30, 30, 30],
    'foo': [11, 12, 13, 21, 22, 23, 31, 32, 33],
    })
    assert f2.equal(expected)


def test_aggregates():
    # Basic aggregation
    f2 = f1.select(
        'x',
        (min, 'y'),
    )
    expected = Frame({
        'x': [1, 2, 3],
        'y' : [10, 10, 10],
    })
    assert f2.equal(expected)

    # Aggregation and aliasing
    f2 = f1.select(
        'x',
        (max, 'z', 'foo'),
    )
    expected = Frame({
        'x': [1, 2, 3],
        'foo' : [31, 32, 33],
    })
    assert f2.equal(expected)
