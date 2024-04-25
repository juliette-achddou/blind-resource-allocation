# general utils
import math
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
    changes to a string to make it usable inside names of files and figures
    :param string:
    :return:
    """
    s = string.replace(" ", "_")
    s = s.replace(".", "Pt")
    s = s.replace(",", "Cma")
    return s


def random_unit(n):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x * x for x in components))
    v = [x / r for x in components]
    return v
