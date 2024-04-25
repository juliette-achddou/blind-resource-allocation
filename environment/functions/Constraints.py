import numpy as np


class Constraints:
    def __init__(self, dim, matrix_constr, l, u):
        """
        init
        :param dim (int): dimension of the problem
        :param matrix_constr (np.array): matrix A where: l <=A.T X < U defines the constrained domain
        :param l (np.array): vector of lower bounds for A.T X
        :param u (np.array): vector of upper bounds for bound for A.T X
        """
        self.dim = dim
        self.matrix_constr = matrix_constr
        self.l = l
        self.u = u

    def in_constraints(self, x):
        """
        Checks if x is inside the constrained domain
        :param x (np.array): vector
        :return: boolean : True if x is inside the constrained domain, False otherwise
        """
        ext_x = x
        return all(self.matrix_constr.dot(x) - self.l >= 0) & all(self.matrix_constr.dot(x) - self.u <= 0)

    def borders(self, x):
        ext_x = np.append(x, [1])
        return any(self.matrix_constr.dot(ext_x) - self.l == 0) or any(self.matrix_constr.dot(ext_x) - self.u == 0)
