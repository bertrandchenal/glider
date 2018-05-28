from numpy import array, unique, concatenate, meshgrid, ndarray


def to_array(x):
    return x if isinstance(x, ndarray) else array(x)


class Frame:

    def __init__(self, data):
        self.data = {k: to_array(v) for k, v in data.iteritems()}

    def mask(self, mask_ar):
        data = {name: self.data[name][mask_ar] for name in self.data}
        return Frame(data)

    def groupby(self, name):
        col = self.data[name]
        keys, inv = unique(col, return_inverse=True)
        for pos, key in enumerate(keys):
            yield key, self.mask(inv == pos)

    def join(self, other, name):
        ar_left = self.data[name]
        ar_right = other.data[name]
        #Convert values to integer (only needed for non-numeric columns)
        ar_all = concatenate([ar_left, ar_right])
        bins, idx = unique(ar_all, return_inverse=True)
        idx_l = idx[:len(ar_left)]
        idx_r = idx[len(ar_left):]
        # Keep matching combinations
        mg_l, mg_r = meshgrid(idx_l, idx_r, sparse=True)
        mg_mask = mg_l == mg_r
        keep_r, keep_l = mg_mask.nonzero()
        # return self.mask(keep_l), other.mask(keep_r)
        return self.mask(keep_l), other.mask(keep_r)

    def __len__(self):
        return len(self.data[self.data.keys()[0]])

    def __str__(self):
        return '\n'.join(
            '%s -> %s' % (k, ', '.join(str(v) for v in vals[:30])[:30] + '...')
            for k, vals in self.data.items()
        )
