from numpy import (array, unique, concatenate, meshgrid, ndarray,
                   set_printoptions, any, lexsort, nan, take, isclose)


__version__ = '0.0'
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

    def select(self, *cols):
        '''
        each col can be a scalar or one of the following form:
        - (agg, name, alias)
        - (agg, array, alias)
        - (agg, name)
        - (array, alias)
        - (name, alias)
        '''
        data = {}
        aggregates = {}
        # Args can be a scalar or a tuple
        for col in cols:
            if isinstance(col, tuple):
                # Extract tuple
                agg = None
                if len(col) == 3:
                    agg, values, alias = col
                elif callable(col[0]):
                    agg, values = col
                    alias = values
                else:
                    values, alias = col
                # Keep col from self if values is not a collection
                if not isinstance(values, (ndarray, list, tuple)):
                    values = self.data[values]
                # Keep track of aggregates
                if agg:
                    aggregates[alias] = agg
            else:
                alias = col
                values = self.data[col]
            data[alias] = values

        fr = Frame(data)
        if not aggregates:
            return fr

        # Call group by and apply aggregates
        res = []
        on = [c for c in fr.data if c not in aggregates]
        for key, chunk in fr.groupby(*on):
            data = {col: [k] for col, k in zip(on, key)}
            data.update(
                (col, [agg(chunk[col])])
                for col, agg in aggregates.items())
            res.append(Frame(data))
        return Frame.union(*res)

    def head(self, length=10):
        data = {name: arr[:length] for name, arr in self.data.items()}
        return Frame(data)

    def tail(self, length=10):
        data = {name: arr[-length:] for name, arr in self.data.items()}
        return Frame(data)

    def unique(self, *names):
        cols = array([self.data[n] for n in names]).T
        _, idx = unique(cols, return_index=True, axis=0)
        return self.mask(idx)

    def groupby(self, *names):
        # XXX aggregates -> can be handled with a select:
        # select(sum('y'), first('z')) can trigger an implicit groupby
        # on x
        cols = array([self.data[n] for n in names]).T
        keys, inv= unique(cols, return_inverse=True, axis=0)
        for pos, key in enumerate(keys):
            yield key, self.mask(inv == pos)

    def pivot(self, what, by, agg=None):
        # We group by all the columns that are not `what` or `by`
        idx_cols = [c for c in self.data if c not in (what, by)]
        res = self.select(*idx_cols).unique(*idx_cols)
        max_length = len(res)
        for key, fr in self.groupby(by):
            new_col = key[0]
            values = fr[what]
            fr = fr.select(*idx_cols, (values, new_col))
            if agg is not None:
                fr = fr.select(*idx_cols, (agg, new_col))

            res = res.join(fr, *idx_cols)
            if agg is None and len(res) > max_length:
                raise ValueError('Duplicated rows over index columns '
                                 'and no aggregation defined')
        return res

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, values):
        self.data[key] = values

    def join(self, other, *names, how='left'):
        if not names:
            names = list(self.data)
        ar_left = array([self.data[n] for n in names]).reshape(-1, len(names))
        ar_right = array([other.data[n] for n in names]).reshape(-1, len(names))

        #Convert values to integer (only needed for non-numeric columns)
        ar_all = concatenate([ar_left, ar_right], axis=0)
        bins, inv = unique(ar_all, return_inverse=True, axis=0)
        inv_l = inv[:len(ar_left)]
        inv_r = inv[len(ar_left):]
        # Keep matching combinations
        mg_l, mg_r = meshgrid(inv_l, inv_r, sparse=True)
        mg_mask = mg_l == mg_r
        keep_r, keep_l = mg_mask.nonzero()

        left_cols = list(self.data)
        right_cols = [n for n in other.data if n not in left_cols]

        # Inner join:
        inner_right = other.mask(keep_r)
        inner_left = self.mask(keep_l)
        aliases = ((inner_right[rc], rc) for rc in right_cols)
        left_cols.extend(aliases)
        res = inner_left.select(*left_cols)
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

    def equal(self, other):
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        for c in self.data:
            if not isclose(self.data[c], other.data[c], equal_nan=True).all():
                return False
        return True

    def diff(self, other):
        pass  # TODO

    def __str__(self):
        return '\n'.join('%s -> %s' % (k, str(vals))
                         for k, vals in self.data.items())
