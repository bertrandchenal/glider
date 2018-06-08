from numpy import (array, unique, concatenate, meshgrid, ndarray, array_str,
                   set_printoptions, any, lexsort, nan, take)


set_printoptions(threshold=50, edgeitems=10)


class Frame:

    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = {
            k: v if isinstance(v, ndarray) else array(v)
            for k, v in data.items()}

    def mask(self, mask_ar):
        data = {name: self.data[name][mask_ar] for name in self.data}
        return Frame(data)

    def select(self, *names, **aliases):
        data = {name: self.data[name] for name in names}
        for alias, col in aliases.items():
            if not isinstance(col, ndarray):
                col = self.data[col]
            data[alias] = col
        return Frame(data)

    def head(self, length=10):
        data = {name: arr[:length] for name, arr in self.data.items()}
        return Frame(data)

    def tail(self, length=10):
        data = {name: arr[-length:] for name, arr in self.data.items()}
        return Frame(data)

    def groupby(self, *names):
        # XXX aggregates ?
        cols = array([self.data[n] for n in names]).T
        keys, idx = unique(cols, return_inverse=True, axis=0)
        for pos, key in enumerate(keys):
            yield key, self.mask(idx == pos)

    def __getitem__(self, key):
        return self.data[key]

    def join(self, other, *names, how='left'):
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

        left_cols = list(self.data)
        right_cols = [n for n in other.data if n not in left_cols]

        # Inner join:
        inner_right = other.mask(keep_r)
        inner_left = self.mask(keep_l)
        kw = {rc: inner_right[rc] for rc in right_cols}
        res = inner_left.select(*left_cols, **kw)
        if how == 'inner':
            return res

        extra = []
        if how in ('left', 'outer'):
            left_only = (~any(mg_mask, axis=0)).nonzero()
            extra.append(self.mask(left_only))
        if how in ('right', 'outer'):
            right_only = (~any(mg_mask, axis=1)).nonzero()
            cols = list(names) + right_cols
            extra.append(other.mask(right_only).select(*cols))

        return Frame.union(res, *extra)

    @classmethod
    def union(cls, *frames):
        if not frames:
            return Frame()

        # Prepare sorted set of columns
        first = frames[0]
        cols = list(first.data)
        for f in frames:
            cols.extend(c for c in f.data if f not in cols)

        res = Frame({c: [] for c in cols})
        res.append(*frames)
        return res

    def sorted(self, *names):
        '''
        Return a sorted copy of self.
        '''
        idx = lexsort([self.data[n] for n in reversed(names)])
        return self.mask(idx)

    def sort(self, *names):
        '''
        In-place sort
        '''
        idx = lexsort([self.data[n] for n in reversed(names)])
        for col in self.data.values():
            take(col, idx, out=col)

    def append(self, *others):
        '''
        In-place union of self and other frames
        '''
        for other in others:
            for col in self.data:
                if col in other.data:
                    other_col = other.data[col]
                else:
                    other_col = [nan] * len(other)
                self.data[col] = concatenate([self.data[col], other_col])

    def copy(self):
        data = {name: array(self.data[name], copy=True) for name in self.data}
        return Frame(data)

    def __len__(self):
        k = next(iter(self.data.keys()), None)
        if k is None:
            return 0
        return len(self.data[k])

    def __str__(self):
        return '\n'.join('%s -> %s' % (k, str(vals))
                         for k, vals in self.data.items())
