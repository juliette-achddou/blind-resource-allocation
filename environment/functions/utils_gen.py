import re

import numpy as np


def string_to_latex(string):
    """
    changes to a string to make it printable in latex without errors
    """

    s = string.replace("_", r"\_")
    if "init" in s:
        i = s.find("init")
        j = s[i:].find(", ")
        s = s[:i] + s[i + j + 2:]
    s = s.replace("rho", r'$\rho$')
    s = s.replace("theta", r'$\theta$')
    s = s.replace(r"alpha\_0", r'$\alpha_0$')
    s = s.replace("v1", r'$\nu_1$')
    # s = s.replace("init", '$x$')
    # raw_str = s.encode('unicode_escape').decode()
    print(s)
    s = re.sub('init.* alpha', '', s)
    print(s)
    # raw_str = raw_str.replace("alpha", 'alpha_0')
    # raw_str = raw_str.replace("x", 'x0')

    return s


def string_to_name(string):
    """
    changes to a string to include it in the naming of files
    :param string:
    :return:
    """
    s = string.replace(" ", "_")
    s = s.replace(".", "Pt")
    s = s.replace(",", "Cma")
    return s


def constraint_simplex_d(dim):
    """
    outputs the matrix A and the vector of lower bounds l and upper bounds u
    that define a simplex of dimension dim through l <A.T X< u
    :param dim: dimension of the simplex
    :return: A, l, u defined as above
    """
    Ai = np.zeros((dim + 1, dim))
    for i in range(dim):
        Ai[i, i] = 1
    Ai[dim, :] = np.ones(dim)
    l = np.zeros(dim + 1)
    u = np.ones(dim + 1)
    return Ai, l, u

# if __name__ == "__main__":
#     r = constraint_simplex_d(3)
#     print(r)
