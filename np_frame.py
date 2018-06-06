from numpy import array, unique, concatenate, meshgrid, ndarray


def to_array(x):
    return x if isinstance(x, ndarray) else array(x)


class Frame:

    def __init__(self, data):
        self.data = {k: to_array(v) for k, v in data.items()}

    def mask(self, mask_ar):
        data = {name: self.data[name][mask_ar] for name in self.data}
        return Frame(data)

    def select(self, *names):
        data = {name: self.data[name] for name in names}
        return Frame(data)

    def groupby(self, *names):
        # XXX aggregates ?
        cols = array([self.data[n] for n in names]).reshape(-1, len(names))
        keys, idx = unique(cols, return_inverse=True, axis=0)
        for pos, key in enumerate(keys):
            yield key, self.mask(idx == pos)

    def join(self, other, *names):
        if not names:
            names = list(self.data)
        ar_left = array([self.data[n] for n in names]).reshape(-1, len(names))
        ar_right = array([other.data[n] for n in names]).reshape(-1, len(names))

        #Convert values to integer (only needed for non-numeric columns)
        ar_all = concatenate([ar_left, ar_right], axis=0)
        bins, idx = unique(ar_all, return_inverse=True, axis=0)
        idx_l = idx[:len(ar_left)]
        idx_r = idx[len(ar_left):]
        # Keep matching combinations
        mg_l, mg_r = meshgrid(idx_l, idx_r, sparse=True)
        mg_mask = mg_l == mg_r
        keep_r, keep_l = mg_mask.nonzero()
        # return self.mask(keep_l), other.mask(keep_r)
        return self.mask(keep_l), other.mask(keep_r)

    def __len__(self):
        k = next(iter(self.data.keys()))
        return len(self.data[k])

    def __str__(self):
        return '\n'.join(
            '%s -> %s' % (k, ', '.join(str(v) for v in vals[:30])[:30] + '...')
            for k, vals in self.data.items()
        )
