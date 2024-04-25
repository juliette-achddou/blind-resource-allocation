import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy import optimize as optimize

from environment.functions.Constraints import Constraints
from environment.functions.Functions import Functions

current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=2)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('font', weight='bold')
# matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath} \boldmath"]
styles = ['o', '^', 's', 'D', 'p', 'v', '*', 'o', '+', "H", "x", "<", ">"]
colors = current_palette[0:20]
hatch = ["iterate", "\\", "//"]
from .utils_gen import *


class realistic_2d_function(Functions):
    def __init__(self, coeff=20):
        """
        init
        :param coeff: the functions corresponding to each dimension are
                        of the form alpha log(1 + c x)/ log(1 + c). coeff plays the role of c
        """
        self.dim = 2
        self.constraints = Constraints(2, *constraint_simplex_d(2))
        self.coeff = coeff

    def make_name_label(self):
        """
        outputs a name, that will identify the function in the title of the files and plots
        and a label that will be used as a label in the plots
        :return: name, label that correspond to the above description
        """
        inter = "realistic 2d function"" coeff= " + str(self.coeff)
        label = string_to_latex(inter)
        name = string_to_name(inter)
        return name, label

    def dim(self):
        """
        outputs the dimension of the function
        :return:dimension
        """
        return self.dim()

    def base_fun(self, x):
        return np.log(1 + self.coeff * x) / np.log(self.coeff + 1)

    def function_1(self, x):
        """
        function associated to the first covariate
        :param x (float): input
        :return: f_1(x)
        """
        return self.base_fun(x)

    def function_2(self, x):
        """
        function associated to the second covariate
        :param x (float): input
        :return: f_2(x)
        """
        return 0.15 * self.base_fun(x)

    def function_3(self, x):
        """
        function associated to the third covariate
        :param x (float): input
        :return:
        """
        return 0.3 * self.base_fun(x)

    def evaluate(self, x):
        """
        evaluates the  sum of f_i(x_i)
        :param x: vector x
        :return:
        """
        return self.function_1(x[0]) + self.function_2(x[1]) + self.function_3(1 - x[1] - x[0])

    def evaluate_2(self, x, y):
        """
        evaluates the function sum of f_i(x_i), with x_1 = x, x_2 = y.
        The difference with the former function is just the form of the input (vector vs two floats)
        and that if outside of the constraints, the return is - infty
        This function is just needed for the plots
        :param x: float x
        :param y: float y
        :return: sum of f_i(x_i)
        """
        X = np.array([x, y])
        if not self.constraints.in_constraints(X):
            return - np.inf
        else:
            return self.evaluate(X)

    def opti(self):
        """
        outputs the argmax and the max of the function defined by sum of f_i(x_i)
        :return:
        """

        def goal_fun(x):
            return - self.evaluate(x)

        # we use scipy optimize with linear constraits, which are defined by L
        L = scipy.optimize.LinearConstraint(A=np.array([[1, 0], [0, 1], [1, 1]]), lb=np.array([0, 0, 0]),
                                            ub=np.array([1, 1, 1]))
        argmax = scipy.optimize.minimize(goal_fun, [0.2, 0.2], args=(), method='COBYLA', constraints=L, tol=None,
                                         callback=None,
                                         options={'rhobeg': 0.05, 'maxiter': 100000, 'disp': False, 'catol': 0.0002})
        max = self.evaluate(np.array(argmax.x))
        return argmax.x, max

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
        return argmax.x, max

    def plot_2D(self):
        """
        plots (1) the simplex
                (2) the contour lines of the function

        saves the figure
        """
        ## meshgrid with step delta
        delta = 0.01
        x = np.arange(-1, 1, delta)
        y = np.arange(-1, 1, delta)
        X, Y = np.meshgrid(x, y)
        Z = np.fromiter(map(self.evaluate_2, X.ravel(), Y.ravel()), X.dtype).reshape(X.shape)

        # change of basis to plot the simplex as an equilateral triangle
        Y_n = np.sqrt(3) / 2 * Y
        X_n = 1 - x - Y_n * (1 / np.sqrt(3))

        fig, ax = plt.subplots()

        # compute the max and min to know at which levels to plot the contours
        max_ = self.opti()[1]
        min_ = self.opti_min()[1]

        # contour plots
        CS = ax.contour(X_n, Y_n, Z, levels=min_ + np.arange(0.1, 1.1, 0.1) * (max_ - min_))
        ax.clabel(CS, inline=1, fontsize=10, )

        # labels and such
        name, label = self.make_name_label()
        ax.set_title('contour line of ' + label)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # save the figure
        plt.savefig("figures/function_" + name + ".pdf")

    def plot_2D_fig(self, ax):
        """
        plots (1) the simplex
                (2) the contour lines of the function
        on given axes (allows to superimpose figures)
        """
        ## meshgrid with step delta
        delta = 0.01
        x = np.arange(0, 1, delta)
        y = np.arange(0, 1, delta)
        print("evalu_min", self.evaluate_2(0, 1))
        X, Y = np.meshgrid(x, y)
        Z = np.fromiter(map(self.evaluate_2, X.ravel(), Y.ravel()), X.dtype).reshape(X.shape)

        # change of basis  to plot the simplex as an equilateral triangle
        Y_n = np.sqrt(3) / 2 * Y
        X_n = 1 - x - Y_n * (1 / np.sqrt(3))

        # compute the max and min to know at which levels to plot the contours
        max_ = self.opti()[1]
        print("max_", max_)
        min_ = self.opti_min()[1]
        print("min_", min_)

        # contour plots
        CS = ax.contour(X_n, Y_n, -Z, levels=-0.6 + (-max_ + 0.6) * np.arange(4 / 5, 0, -1 / 5), alpha=0.5,
                        colors='darkorange')
        ax.clabel(CS, inline=1, fontsize=10, colors=['darkorange'])

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # just plot the figure, do not save it


if __name__ == "__main__":
    r = realistic_2d_function()
    r.opti()
