import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from environment.Noise.gaussian_Noise import GaussianNoise
from environment.functions.realistic_2d_function import realistic_2d_function
from environment.functions.realistic_5d_function import realistic_5d_function
from environment.functions.realistic_6d_function import realistic_6d_function
from environment.functions.real_2d_function_opti_border import realistic_2d_function_border

from environment.game import game
from policy.HOOConstrainedSimplex import HOOConstrainedSimplex
from policy.UCBDiscrete import UCBDiscrete
from policy.fds_plan import FDS_Plan
from policy.fds_seq import FDS_Seq
from policy.onepointestimationdescent import OnePointEstimationDescent
from policy.twopointsestimationdescent import TwoPointsEstimationDescent
from utils_experiments import parallel_experiments_compare_plots_save, normalize, \
    get_directions_from_dimension

sns.set()

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


def main():

    ##1_ Dimension 2 FOR ILLUSTRATION

    # environment
    dim = 2
    sigma = 0.1
    noise = GaussianNoise(sigma)

    coeff = 8
    fun = realistic_2d_function(coeff)
    fun.plot_2D()

    env = game(fun, noise)

    # parameters of the experiment
    nbRep = 1
    n_pool = 1
    horizon = 100000

    # parameters for policies
    theta = 0.7
    c = 1
    alpha0 = 0.2
    D_k = get_directions_from_dimension(dim)
    shuffle = False

    # policies
    policy_UCB = UCBDiscrete(horizon=horizon, dim=dim, constraints=fun.constraints, sigma=sigma)
    policy_FDS_Plan = FDS_Plan(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                               constraints=fun.constraints, D_k=D_k, sigma=sigma, delta=(horizon) ** (-4 / 3) / 2,
                               shuffle=shuffle)
    policy_FDS_Seq = FDS_Seq(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                             constraints=fun.constraints, D_k=D_k, sigma=sigma,
                             delta=(horizon) ** (-10 / 3) / 2, shuffle=shuffle)
    policy_HOO = HOOConstrainedSimplex(v1=2 * coeff, prec=1 / np.sqrt(horizon), constraints=fun.constraints, dim=dim,
                                       sigma=sigma, horizon=horizon)
    policy_GD1 = OnePointEstimationDescent(dim=dim, init_point=[1 / 3, 1 / 3],
                                           center_point=[(np.sqrt(2) - 1) / (np.sqrt(2)),
                                                         (np.sqrt(2) - 1) / (np.sqrt(2))],
                                           radius=1 / (2 + np.sqrt(2)), constraints=fun.constraints, alpha=2.5)
    policy_GD2 = TwoPointsEstimationDescent(dim=dim, init_point=[1 / 3, 1 / 3],
                                            center_point=[(np.sqrt(2) - 1) / (np.sqrt(2)),
                                                          (np.sqrt(2) - 1) / (np.sqrt(2))],
                                            radius=1 / (2 + np.sqrt(2)), constraints=fun.constraints, alpha=2.5)
    policy_FDS_Plan_D = FDS_Plan(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                                 constraints=fun.constraints, sigma=sigma, delta=(horizon) ** (-4 / 3) / 2,
                                 shuffle=shuffle)
    policy_FDS_Seq_D = FDS_Seq(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                               constraints=fun.constraints, sigma=sigma, delta=1 / horizon)

    policies = [policy_UCB, policy_FDS_Plan, policy_FDS_Seq, policy_HOO, policy_GD1, policy_GD2, policy_FDS_Plan_D,
                policy_FDS_Seq_D]

    #
    name_dir = "1_exp_"
    # parallel_experiments_compare_plots(env, policies, n_pool, horizon, name_dir, nbRep, contour=True, split=False)
    # # parallel_experiments_compare_plots_save(env,policies,n_pool,horizon,name_dir, nbRep, contour = True, split = False)



    ## OPTI ON THE BORDER

    # environment
    dim = 2
    sigma = 0.1
    noise = GaussianNoise(sigma)

    coeff = 8
    fun = realistic_2d_function_border(coeff)
    fun.plot_2D()

    env = game(fun, noise)

    # parameters of the experiment
    nbRep = 1
    n_pool = 1
    horizon = 100000

    # parameters for policies
    theta = 0.7
    c = 1
    alpha0 = 0.2
    D_k = get_directions_from_dimension(dim)
    shuffle = False

    # policies
    policy_UCB = UCBDiscrete(horizon=horizon, dim=dim, constraints=fun.constraints, sigma=sigma)
    policy_FDS_Plan = FDS_Plan(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                               constraints=fun.constraints, D_k=D_k, sigma=sigma, delta=(horizon) ** (-4 / 3) / 2,
                               shuffle=shuffle)
    policy_FDS_Seq = FDS_Seq(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                             constraints=fun.constraints, D_k=D_k, sigma=sigma,
                             delta=(horizon) ** (-10 / 3) / 2, shuffle=shuffle)
    policy_HOO = HOOConstrainedSimplex(v1=2 * coeff, prec=1 / np.sqrt(horizon), constraints=fun.constraints, dim=dim,
                                       sigma=sigma, horizon=horizon)
    policy_GD1 = OnePointEstimationDescent(dim=dim, init_point=[1 / 3, 1 / 3],
                                           center_point=[(np.sqrt(2) - 1) / (np.sqrt(2)),
                                                         (np.sqrt(2) - 1) / (np.sqrt(2))],
                                           radius=1 / (2 + np.sqrt(2)), constraints=fun.constraints, alpha=2.5)
    policy_GD2 = TwoPointsEstimationDescent(dim=dim, init_point=[1 / 3, 1 / 3],
                                            center_point=[(np.sqrt(2) - 1) / (np.sqrt(2)),
                                                          (np.sqrt(2) - 1) / (np.sqrt(2))],
                                            radius=1 / (2 + np.sqrt(2)), constraints=fun.constraints, alpha=2.5)
    policy_FDS_Plan_D = FDS_Plan(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                                 constraints=fun.constraints, sigma=sigma, delta=(horizon) ** (-4 / 3) / 2,
                                 shuffle=shuffle)
    policy_FDS_Seq_D = FDS_Seq(alpha0=alpha0, init_point=[1 / 3, 1 / 3], theta=theta, gamma=1., c=c, K=2,
                               constraints=fun.constraints, sigma=sigma, delta=1 / horizon)

    policies = [policy_UCB, policy_FDS_Plan, policy_FDS_Seq, policy_HOO, policy_GD1, policy_GD2, policy_FDS_Plan_D,
                policy_FDS_Seq_D]

    #
    name_dir = "bord_exp_"
    # parallel_experiments_compare_plots(env, policies, n_pool, horizon, name_dir, nbRep, contour=True, split=False)
    # parallel_experiments_compare_plots_save(env,policies,n_pool,horizon,name_dir, nbRep, contour = True, split = False)

    # ########################################################################################################################
    # ## 3_ random shift FOR REGRET PLOTS





    #### Dimension 5
    dim = 5
    sigma = 0.1
    noise = GaussianNoise(sigma)
    coeff = 8
    fun = realistic_5d_function(coeff)
    env = game(fun, noise, randomization=True, randomization_step=0.05)

    # parameters for policies
    init_dis = 0.01
    theta = 0.8
    c = 1
    alpha0 = 0.4
    D_k = normalize(get_directions_from_dimension(dim))

    # parameters for the experiment
    nbRep = 50 * 24
    n_pool = 24

    horizon = 500000
    policy_UCB_5 = UCBDiscrete(horizon=horizon, dim=dim, constraints=fun.constraints, sigma=sigma)

    policy_FDS_Plan_D_5 = FDS_Plan(alpha0=alpha0, init_point=init_dis * np.ones(dim), theta=theta, gamma=1., c=c, K=2,
                                   constraints=fun.constraints, sigma=sigma, delta=1 / horizon)
    policy_FDS_Seq_D_5 = FDS_Seq(alpha0=alpha0, init_point=init_dis * np.ones(dim), theta=theta, gamma=1., c=c,
                                 K=2,
                                 constraints=fun.constraints, sigma=sigma, delta=1 / horizon)
    policy_GD2_5 = TwoPointsEstimationDescent(dim=dim, init_point=init_dis * np.ones(dim),
                                              center_point=1 / (dim + 1) * np.ones(dim),
                                              radius=1 / (4 * np.sqrt(dim * (dim + 1))), constraints=fun.constraints,
                                              alpha=2.5)
    policies = [policy_UCB_5, policy_FDS_Seq_D_5, policy_GD2_5, policy_FDS_Plan_D_5]

    name_dir = "exp5_01_"

    # parallel_experiments_compare_plots(env, policies, n_pool, horizon, name_dir, nbRep, tsav = 100* np.arange(int(horizon/100)))
    # parallel_experiments_compare_plots_save(env,policies,n_pool,horizon,name_dir, nbRep, ylim_top = 10000)
    print("opti", fun.opti())

    #### Dimension 6
    dim = 6
    sigma = 0.1
    noise = GaussianNoise(sigma)
    coeff = 8
    fun = realistic_6d_function(coeff)
    env = game(fun, noise, randomization=True, randomization_step=0.05)

    # parameters for policies
    init_dis = 0.01
    theta = 0.8
    c = 1
    alpha0 = 0.4
    D_k = normalize(get_directions_from_dimension(dim))

    # parameters for the experiment
    nbRep = 20 * 24
    n_pool = 24

    horizon = 500000
    policy_UCB_6 = UCBDiscrete(horizon=horizon, dim=dim, constraints=fun.constraints, sigma=sigma)

    policy_FDS_Plan_D_6 = FDS_Plan(alpha0=alpha0, init_point=init_dis * np.ones(dim), theta=theta, gamma=1., c=c, K=2,
                                   constraints=fun.constraints, sigma=sigma, delta=1 / horizon)
    policy_FDS_Seq_D_6 = FDS_Seq(alpha0=alpha0, init_point=init_dis * np.ones(dim), theta=theta, gamma=1., c=c,
                                 K=2,
                                 constraints=fun.constraints, sigma=sigma, delta=1 / horizon)
    policy_GD2_6 = TwoPointsEstimationDescent(dim=dim, init_point=init_dis * np.ones(dim),
                                              center_point=1 / (dim + 1) * np.ones(dim),
                                              radius=1 / (4 * np.sqrt(dim * (dim + 1))), constraints=fun.constraints,
                                              alpha=2.5)
    policies = [policy_UCB_6, policy_FDS_Seq_D_6, policy_GD2_6, policy_FDS_Plan_D_6]

    name_dir = "exp6_01_"

    # parallel_experiments_compare_plots(env, policies, n_pool, horizon, name_dir, nbRep, tsav = 100* np.arange(int(horizon/100)))
    # parallel_experiments_compare_plots_save(env,policies,n_pool,horizon,name_dir, nbRep,tsav = 100* np.arange(int(horizon/100)),  ylim_top = 15000)
    print("opti", fun.opti())


if __name__ == "__main__":
    main()
