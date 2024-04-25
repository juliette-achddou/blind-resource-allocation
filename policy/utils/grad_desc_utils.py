import numpy as np


def proj(a, y):
    """
    projection onto a simplex of y. a is supposed to be a = (1,1, ... 1)
    http://www.mcduplessis.com/index.php/2016/08/22/fast-projection-onto-a-simplex-python/
    :param a: (1,1, ... 1)
    :param y: vector to project
    :return: projection
    The idea is that lambda in the Lagrangian is one of (sum_{i =1}^n ai yi +1)/ (sum_{i =1}^k ai yi +1), for some k.
    We find the right k thanks a bisection search. Then we compte the associated x
    """
    li = y / a
    idx = np.argsort(li)
    d = len(li)
    evalpL = lambda k: np.sum(a[idx[k:]] * (y[idx[k:]] - li[idx[k]] * a[idx[k:]])) - 1

    def bisectsearch():
        idxL, idxH = 0, d - 1
        L = evalpL(idxL)
        H = evalpL(idxH)
        if L < 0:
            return idxL
        while (idxH - idxL) > 1:
            iMid = int((idxL + idxH) / 2)
            M = evalpL(iMid)
            if M > 0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M
        return idxH

    k = bisectsearch()
    lam = (np.sum(a[idx[k:]] * y[idx[k:]]) - 1) / np.sum(a[idx[k:]])
    x = np.maximum(0, y - lam * a)
    return x


def projection_simplex(x, d):
    """
    projection onto a simplex of x completed by 1 - sum x_i.
    :param x: point (described by its first d coordinates)
    :param d: dimension
    :return: projection
    """
    a = np.ones(d + 1)
    y = np.append(x, [1 - x.sum()])
    pr = proj(a, y)

    return pr[:-1]
