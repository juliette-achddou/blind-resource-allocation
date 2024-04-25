import scipy
from scipy import optimize as optimize

from environment.functions.Constraints import Constraints
from environment.functions.Functions import Functions
from .utils_gen import *


class realistic_3d_function(Functions):
    def __init__(self, coeff=20):
        """
        init
        :param coeff: the functions corresponding to each dimension are
                        of the form alpha log(1 + c x)/ log(1 + c). coeff plays the role of c
        """
        self.dim = 3
        # print("hey3d")
        self.constraints = Constraints(3, *constraint_simplex_d(3))
        self.coeff = coeff

    def make_name_label(self):
        """
        outputs a name, that will identify the function in the title of the files and plots
        and a label that will be used as a label in the plots
        :return: name, label that correspond to the above description
        """
        inter = "realistic 3d function" + " coeff= " + str(self.coeff)
        label = string_to_latex(inter)
        name = string_to_name(inter)
        return name, label

    def dim(self):
        """
        outputs the dimension of the function
        :return:dimension
        """
        return self.dim()

    def function_1(self, x):
        """
        function associated to the first covariate
        :param x (float): input
        :return: f_1(x)
        """
        return np.log(1 + self.coeff * x) / np.log(self.coeff + 1)

    def function_2(self, x):
        """
        function associated to the second covariate
        :param x (float): input
        :return: f_2(x)
        """
        return 0.45 * np.log(1 + self.coeff * x) / np.log(self.coeff + 1)

    def function_3(self, x):
        """
        function associated to the third covariate
        :param x (float): input
        :return: f_3(x)
        """
        return 0.87 * np.log(1 + self.coeff * x) / np.log(self.coeff + 1)

    def function_4(self, x):
        """
        function associated to the fourth covariate
        :param x (float): input
        :return: f_4(x)
        """
        return 0.95 * np.log(1 + self.coeff * x) / np.log(self.coeff + 1)

    def evaluate(self, x):
        """
        evaluates the  sum of f_i(x_i)
        :param x: vector x
        :return: sum of f_i(x_i)
        """
        return self.function_1(x[0]) + self.function_2(x[1]) + self.function_3(x[2]) + self.function_4(
            1 - x[1] - x[0] - x[2])

    def opti_min(self):
        """
        outputs the argmin and the min of the function defined by sum of f_i(x_i)
        :return:
        """

        def goal_fun(x):
            return self.evaluate(x)

        L = scipy.optimize.LinearConstraint(A=np.array([[1, 0], [0, 1], [1, 1]]), lb=np.array([0, 0, 0]),
                                            ub=np.array([1, 1, 1]))
        argmax = scipy.optimize.minimize(goal_fun, [0.2, 0.2], args=(), method='COBYLA', constraints=L, tol=None,
                                         callback=None,
                                         options={'rhobeg': 0.05, 'maxiter': 100000, 'disp': False, 'catol': 0.0002})
        max = self.evaluate(np.array(argmax.x))
        print(argmax, max)

    def opti(self):
        """
        outputs the argmax and the max of the function defined by sum of f_i(x_i)
        :return: the argmax and the max
        """

        # argmax = np.array([0.30087, 0.160526, 0.255263 ])
        # max = self.evaluate(argmax)
        def goal_fun(x):
            return - self.evaluate(x)

        L = scipy.optimize.LinearConstraint(A=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]),
                                            lb=np.array([0, 0, 0, 0]), ub=np.array([1, 1, 1, 1]))
        argmax = scipy.optimize.minimize(goal_fun, [0.2, 0.2, 0.2], args=(), method='COBYLA', constraints=L, tol=None,
                                         callback=None,
                                         options={'rhobeg': 0.05, 'maxiter': 100000, 'disp': False, 'catol': 0.0002})
        # print("here",np.array(argmax))
        max = self.evaluate(np.array(argmax.x))
        print(argmax, max)
        # 0.820728
        return argmax.x, max

# if __name__ == "__main__":
#     r = real_3d_function()
#     r.opti()
