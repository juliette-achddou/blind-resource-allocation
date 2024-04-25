import multiprocessing as mp
import re

import matplotlib
import matplotlib.cm as cmx
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from Evaluation import Evaluation

current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=2)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('font', weight='bold')
# matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{} \boldmath"
styles = ['o', '^', 's', 'D', 'p', 'v', '*', 'o', '+', "H", "x", "<", ">"]
colors = current_palette[0:20]
hatch = ["iterate", "\\", "//"]


def string_to_latex(string):
    """
    Transforms a '_' into a '/\_'.
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
    s = re.sub(r'.*init.*, ', '', s)
    print(s)
    # raw_str = raw_str.replace("alpha", 'alpha_0')
    # raw_str = raw_str.replace("x", 'x0')

    return s

    raw_str = s.encode('unicode_escape').decode()
    raw_str = raw_str.replace("_", "\_")
    print(raw_str)
    return raw_str


def string_to_name(string):
    """changes to a string to make it printable in latex without errors"""
    s = string.replace(" ", "_")
    s = s.replace(".", "Pt")
    s = s.replace(",", "Cma")
    return s


def normalize(vects):
    new_vects = []
    for v in vects:
        new_vects.append(v / np.linalg.norm(v))
    return new_vects


def get_directions_from_dimension(dim):
    res = np.zeros(((2 * (dim)) * (dim), dim + 1))
    for i in range(dim):
        for l in range(2 * dim * i, 2 * dim * i + dim):
            res[l, i] = 1
            z = (l - 2 * dim * i) % (dim)
            if z < i:
                res[l, z] = -1
            else:
                res[l, z + 1] = -1
        for l in range(2 * dim * i + dim, 2 * dim * (i + 1)):
            res[l, :] = -res[l - dim, :]
    res = res[:, :-1]
    res = np.unique(res, axis=0)
    res = list(res)
    return res


def evReward_full_list(list_of_args):
    """
    For each tuple of arguments (env, policy, nbRep, horizon)
    in list_of_args,
    simulates a nbRep sequences of allocation problems in the given environment with the
    given policy and computes the average regret.

    :param list_of_args:  list of arguments (env, policy, nbRep, horizon)
    :type listreg: list(tuples(Environment object, Policy object, int, int))
    :Returns: tuple of the average regret, and number of Monte Cartlo trials
    :rtype: tuple
    """
    # prof = cProfile.Profile()
    env, policy, nbRep, horizon, tsav = list_of_args
    # np.random.seed()
    # print("XXXXXXXXXXXXXXXXXXXXXXdimension", env.constraints.dim)
    # import pdb; pdb.set_trace()
    ev = Evaluation(env, policy, nbRep, horizon,
                    tsav=tsav, print_=False, dim=env.constraints.dim)
    cumRegret = ev.cumRegret
    distance_to_opti = ev.distance_to_opti
    choices = ev.choices
    it_choices = ev.it_choices
    return (cumRegret, nbRep, distance_to_opti, choices, it_choices)


def extended_list(list_reg, n):
    """
    :param list_reg:  list of tuples (regret_List,nbRep,stopping_TimeList)
        number_of_items_aggregated)
    :type list_reg: list(tuples)
    :param n: index of the tuple to be processed
    :type n: int
    """
    extended_list = []
    for l in list_reg:
        extended_list.extend(list(l[n]))
    return extended_list


def single_experiment_plot(env, policy, horizon, figure_numb, label, name, k):
    """
    plot the regret curve and the distance to the optimum corresponding to one experiment
    :param env:
    :param policy:
    :param horizon:
    :param figure_numb:
    :param label:
    :param name:
    :param k:
    :return:
    """
    plt.figure(figure_numb)
    result = env.play(policy, horizon)
    reg = result.getRegret()
    print(reg)
    plt.plot(np.arange(horizon), reg, label=label, color=colors[k])
    dist = result.distance_to_opti
    plt.legend()
    plt.savefig("figures/" + name + ".pdf")
    plt.close()
    plt.figure()
    plt.plot(np.arange(horizon), dist, label=label, color=colors[k])
    plt.legend()
    plt.savefig("figures/" + "distance_" + name + ".pdf")
    plt.close()


def scatter3d(X, cs, colorsMap='jet', name='3d'):
    """
    plots a trajectory in 3D.
    :param X: array with the trajectory: we have x = X[:, 0], y = X[:, 1], z = X[:, 2] are the corrdinates
    :param cs:  colors
    :param colorsMap:
    :param name: name to be included in the name of the file
    :return:
    """
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure("scatter")
    ax = Axes3D(fig)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), alpha=0.2)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)

    # pickle.dump(fig,open('figure2.pickle', 'wb'))  # This is for Python 3 - py2 may need `file` instead of `open`
    plt.savefig(name + ".pdf")
    # plt.show()
    plt.close()
    u, indices = np.unique(X, return_index=True, axis=0)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    ax.scatter(x, y, z, c=scalarMap.to_rgba(indices), alpha=0.7)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.savefig(name + "_first" + ".pdf")
    # plt.show()
    plt.close()


def scatter_plot_2d_original_simplex(X, z, colorsMap='jet', name='2d'):
    """
    Plots the trajectory in the original simplex described by x_t, y_t where we have x = X[:, 0], y = X[:, 1]
    :param X:
    :param z:
    :param colorsMap:
    :param name:
    :return:
    """
    x = X[:, 0]
    y = X[:, 1]

    fig = plt.figure("scatter", figsize=(6, 6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(x, y, c=z, marker='o', cmap=cm.jet, alpha=0.2)
    plt.colorbar(sc)
    plt.savefig(name + ".pdf")
    # plt.show()
    plt.close()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    u, indices = np.unique(X, return_index=True, axis=0)
    x = u[:, 0]
    y = u[:, 1]
    sc = ax.scatter(x, y, c=indices, marker='o', cmap=cm.jet, alpha=0.7)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(sc)
    plt.savefig(name + "_fst" + ".pdf")
    # plt.show()
    plt.close()


def scatter_2d_jet_reshaped_simplex(X, name='2d', name_policy="", title=False, fun=None, contour=False, iterates=None,
                                    logsc=True,
                                    split=True):
    """
    :param X:
    :param name:
    :param name_policy:
    :param title:
    :param fun:
    :param contour:
    :param iterates: if iterates is not None, it means that the policy involves iterates (like those of FDS)
    Then it contains a vector with the trajectory of the those iterates
    :param logsc:
    :param split:
    :return: a plot of the reshaped simplex (equilateral triangle) with the trajectories of the points, and iterates if
    need be. The contourplot of the function is also represented
    """
    global it_x_n
    x = X[:, 0]
    y = X[:, 1]
    T = len(x)

    y_n = np.sqrt(3) / 2 * y
    x_n = 1 - x - y_n * (1 / np.sqrt(3))

    if iterates is not None:
        it_x = iterates[:, 0]
        it_y = iterates[:, 1]
        it_y_n = np.sqrt(3) / 2 * it_y
        it_x_n = 1 - it_x - it_y_n * (1 / np.sqrt(3))
    # x_n = x_n + np.random.normal(loc=0.0, scale=0.01, size=T)
    # y_n = y_n + np.random.normal(loc=0.0, scale=0.01, size=T)
    z = np.arange(0, 1, 0.001)
    z_1 = 1 / 2 * z
    z_2 = 1 / 2 + z_1
    if split:
        fig = plt.figure("scatter", figsize=(20, 4))
        values = [int(T / 100), int(T / 10), int(T)]
        l_b = 131
    else:
        fig = plt.figure("scatter")
        values = [int(T)]
        l_b = 111

    for i, t in enumerate(values):
        l = l_b + i
        ax = fig.add_subplot(l)
        xt = x_n[:t]
        yt = y_n[:t]
        Xt = np.stack((xt, yt), axis=-1)
        u, indices, unique_inverse, unique_counts = np.unique(Xt, return_index=True, return_inverse=True,
                                                              return_counts=True, axis=0)
        if iterates is not None:
            it_xt = it_x_n[:t]
            it_yt = it_y_n[:t]
            it_Xt = np.stack((it_xt, it_yt), axis=-1)
            it_u, it_indices, it_unique_inverse, it_unique_counts = np.unique(it_Xt, return_index=True,
                                                                              return_inverse=True,
                                                                              return_counts=True, axis=0)

        plt.plot(z, np.zeros(len(z)), "r", zorder=1)
        plt.plot(z_1, np.sqrt(3) * z_1, "r", zorder=1)
        plt.plot(z_2, np.sqrt(3) / 2 - np.sqrt(3) * z_1, "r", zorder=1)
        ax.set_xlim(left=-0.05, right=1 + 0.05)
        ax.set_ylim(bottom=-0.05, top=np.sqrt(3) / 2 + 0.05)
        if contour == True and fun != None:
            fun.plot_2D_fig(ax)
        if split:
            ax.set_title("t=" + str(t))
        # print("u",u)
        # print(size, u[:,0].shape,u[:,0].shape )
        # sc = ax.scatter(u[:,0],u[:,1],c = indices, marker = 'o', alpha = 0.9, s = 200*unique_counts/np.max(unique_counts))
        if logsc:
            sc = ax.scatter(u[:, 0], u[:, 1], c=indices + 1,
                            marker='o', alpha=0.9,
                            s=200 * np.log(1 + 0.1 * unique_counts) / np.log(T),
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=T),
                            zorder=2)
            if iterates is not None:
                sc2 = ax.scatter(it_u[:, 0], it_u[:, 1],
                                 marker='o', alpha=0.9,
                                 s=202 * np.log(1 + 0.1 * it_unique_counts) / np.log(T),
                                 facecolors='none', edgecolors='deepskyblue',
                                 zorder=2)

        else:
            sc = ax.scatter(u[:, 0], u[:, 1], c=indices + 1,
                            marker='o', alpha=0.9,
                            s=200 * np.sqrt(unique_counts) / np.sqrt(T),
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=T),
                            zorder=2)
            if iterates != None:
                sc2 = ax.scatter(it_u[:, 0], it_u[:, 1],
                                 marker='o', alpha=0.9,
                                 s=202 * np.sqrt(it_unique_counts) / np.sqrt(T),
                                 facecolors='none', edgecolors='deepskyblue',
                                 zorder=2)
        tick_font_size = 14
        # cbar =plt.colorbar(sc)
        cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.55, pad=0.05)
        cbar.ax.tick_params(labelsize=tick_font_size)
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')

        ar, ma = fun.opti()
        ary = np.sqrt(3) / 2 * ar[1]
        arx = 1 - ar[0] - ary * (1 / np.sqrt(3))
        ax.scatter(arx, ary, marker='x', color="black", zorder=3)
        fig.tight_layout()
        plt.axis('off')
    if title:
        fig.suptitle("Evaluation points of " + name_policy + " at different times")
    # plt.tight_layout()

    plt.savefig(name + "_various_times" + ".pdf")
    plt.close()


def parallel_experiments_plots(env, policy, n_pool, horizon, figure_numb, label, name, k, nbRep=1, colors=colors,
                               tsav=None, styles=styles):
    """
    For one policy, and environment, plots the regret and the distance to the optimum, but also, when possible (d<3)
    the trajectories
    :param env:
    :param policy:
    :param n_pool:
    :param horizon:
    :param figure_numb:
    :param label:
    :param name:
    :param k:
    :param nbRep:
    :param colors:
    :param styles:
    :return:
    """
    if tsav is None:
        tsav = np.arange(horizon)
    pool = mp.Pool(n_pool)
    args = [[env, policy, int(nbRep / n_pool), horizon, tsav]] * n_pool
    # import pdb;pdb.set_trace()
    regret_by_pool = pool.map(evReward_full_list, args)
    regret_list = np.array(extended_list(regret_by_pool, 0))
    distance_list = np.array(extended_list(regret_by_pool, 2))
    choices_list = np.array(extended_list(regret_by_pool, 3))
    meanRegret = regret_list.mean(axis=0)
    meanChoices = choices_list.mean(axis=0)
    quant_0 = np.quantile(regret_list, 0.25, axis=0)
    quant_1 = np.quantile(regret_list, 0.75, axis=0)
    meanDistance = distance_list.mean(axis=0)
    print("meanreg_1 ", meanRegret)
    plt.figure()
    plt.plot(1 + tsav, meanRegret,
             color=colors[k], marker=styles[k], markevery=500, label=label
             )
    plt.fill_between(1 + tsav, quant_0,
                     quant_1,
                     color=colors[k], alpha=0.2)
    plt.ylim(bottom=0)
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig("figures/regret_" + name)
    plt.figure(figure_numb + 1)
    plt.plot(1 + tsav, meanDistance, label=label, color=colors[k])
    # plt.xscale('log')
    plt.legend()
    plt.savefig("figures/" + "distance_" + name)
    plt.close("all")
    if env.constraints.dim == 3:
        scatter3d(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
    if env.constraints.dim == 2:
        scatter_plot_2d_original_simplex(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
    if env.constraints.dim == 1:
        fig = plt.figure("scatter")
        ax = fig.add_subplot(111)
        ax.scatter(tsav + 1, meanChoices[:])
        plt.savefig("figures/" + "choice" + name + ".png")
        # plt.show()
        plt.close("all")


def parallel_experiments_compare_plots(env, policies, n_pool, horizon, name_dir, nbRep=1, colors=colors, styles=styles,
                                       path="figures", contour=False, split=True, tsav=None):
    """
    For multiple policies, and one environment, plots the regret curves on one figure
    and the distances to the optimum on one plot , but also, when possible (d<3), ofr each policy, the trajectory
    the trajectories
    :param env:
    :param policies:
    :param n_pool:
    :param horizon:
    :param name_dir:
    :param nbRep:
    :param colors:
    :param styles:
    :param path:
    :param contour:
    :param split:
    :return:
    """
    if tsav is None:
        tsav = np.arange(horizon)
    pool = mp.Pool(n_pool)
    plt.figure("reg")
    plt.figure("mDis")
    name_f, label_f = env.function.make_name_label()
    name_noise = name_dir + env.noise.__class__.__name__ + "_sigma_" + string_to_name(
        str(env.noise.sigma)) + "_hor_" + str(horizon)
    name_pref = "_".join([name_f, name_noise, "nbRep", string_to_name(str(nbRep))])
    for k, policy in enumerate(policies):
        name_pol, label = policy.make_name_label()
        name = "_".join([name_pref, name_pol])
        print(name)
        args = [[env, policy, int(nbRep / n_pool), horizon, tsav]] * n_pool
        # import pdb;pdb.set_trace()
        regret_by_pool = pool.map(evReward_full_list, args)
        regret_list = np.array(extended_list(regret_by_pool, 0))
        distance_list = np.array(extended_list(regret_by_pool, 2))
        choices_list = np.array(extended_list(regret_by_pool, 3))
        it_choices_list = np.array(extended_list(regret_by_pool, 4))
        print("it_choices_list", it_choices_list)
        with open(''.join([path, "/save/" + name, "regret_list", '.npy']), 'wb') as f:
            np.save(f, regret_list)
        with open(''.join([path, "/save/" + name, "choice_list", '.npy']), 'wb') as f:
            np.save(f, choices_list)
        with open(''.join([path, "/save/" + name, "it_choice_list", '.npy']), 'wb') as f:
            np.save(f, it_choices_list)
        with open(''.join([path, "/save/" + name, "distance_list", '.npy']), 'wb') as f:
            np.save(f, distance_list)

        meanRegret = regret_list.mean(axis=0)
        meanChoices = choices_list.mean(axis=0)
        mean_it_choices = it_choices_list.mean(axis=0)
        quant_0 = np.quantile(regret_list, 0.25, axis=0)
        quant_1 = np.quantile(regret_list, 0.75, axis=0)
        meanDistance = distance_list.mean(axis=0)
        print("meanreg_1 ", meanRegret)
        plt.figure("reg")
        plt.plot(1 + tsav, meanRegret,
                 color=colors[k], marker=styles[k], markevery=int(len(tsav) / 20), label=label
                 )
        plt.fill_between(1 + tsav, quant_0,
                         quant_1,
                         color=colors[k], alpha=0.2)

        plt.figure("mDis")
        plt.plot(1 + tsav, meanDistance, label=label, color=colors[k])
        # plt.xscale('log')
        plt.legend()
        plt.savefig("figures/" + "distance_" + name + ".pdf")

        if env.constraints.dim == 3:
            scatter3d(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
        if env.constraints.dim == 2:
            scatter_plot_2d_original_simplex(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
            if policy.iterate_bool:
                scatter_2d_jet_reshaped_simplex(meanChoices, name="figures/" + "choices_" + name,
                                                name_policy=string_to_latex(policy.__class__.__name__),
                                                fun=env.function,
                                                contour=contour, iterates=mean_it_choices, split=split)
            else:
                scatter_2d_jet_reshaped_simplex(meanChoices, name="figures/" + "choices_" + name,
                                                name_policy=string_to_latex(policy.__class__.__name__),
                                                fun=env.function, contour=contour, split=split)
        if env.constraints.dim == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(tsav + 1, meanChoices[:])
            plt.savefig("figures/" + "choice" + name + ".pdf")
            plt.close()
        # plt.show()
    plt.ylim(bottom=0)
    plt.figure("reg")
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig("figures/comp_regret_" + name_pref + ".pdf")
    plt.figure("mDis")
    # plt.xscale('log')
    plt.legend()
    plt.savefig("figures/comp_distance_" + name_pref + ".pdf")

    plt.close("all")


def parallel_experiments_compare_plots_save(env, policies, n_pool, horizon, name_dir, nbRep=1, colors=colors,
                                            styles=styles, path="figures", title=False, contour=False, split=True,
                                            choice=False, ylim_top=False, tsav=None):
    """ same as above function, but the data is already stored, so the experiments do not need to be done,
    only the plotting"""
    if tsav is None:
        tsav = np.arange(horizon)
    pool = mp.Pool(n_pool)
    fig1 = plt.figure("reg")
    ax1 = fig1.add_subplot(111)
    plt.figure("mDis")
    name_f, label_f = env.function.make_name_label()
    name_noise = name_dir + env.noise.__class__.__name__ + "_sigma_" + string_to_name(
        str(env.noise.sigma)) + "_hor_" + str(horizon)
    name_pref = "_".join([name_f, name_noise, "nbRep", string_to_name(str(nbRep))])
    for k, policy in enumerate(policies):
        name_pol, label = policy.make_name_label()
        name = "_".join([name_pref, name_pol])
        # args = [[env, policy, int(nbRep / n_pool), horizon, tsav]] * n_pool
        # import pdb;pdb.set_trace()
        # regret_by_pool = pool.map(evReward_full_list, args)
        f_reg = (
            ''.join([path, "/save/" + name, "regret_list", '.npy'.replace(" ", "_")]))
        f_dis = (
            ''.join([path, "/save/" + name, "distance_list", '.npy'.replace(" ", "_")]))
        f_choice = (
            ''.join([path, "/save/" + name, "choice_list", '.npy'.replace(" ", "_")]))
        if choice:
            f_it_choice = (
                ''.join([path, "/save/" + name, "it_choice_list", '.npy'.replace(" ", "_")]))
        regret_list = np.load(f_reg)
        # print(regret_list.shape)
        distance_list = np.load(f_dis)
        choices_list = np.load(f_choice)
        if choice:
            it_choices_list = np.load(f_it_choice)

        with open(''.join([path, "/save/" + name, "regret_list", '.npy']), 'wb') as f:
            np.save(f, regret_list)
        with open(''.join([path, "/save/" + name, "choice_list", '.npy']), 'wb') as f:
            np.save(f, choices_list)
        with open(''.join([path, "/save/" + name, "distance_list", '.npy']), 'wb') as f:
            np.save(f, distance_list)
        if choice:
            with open(''.join([path, "/save/" + name, "it_choice_list", '.npy']), 'wb') as f:
                np.save(f, it_choices_list)

        meanRegret = regret_list.mean(axis=0)
        meanChoices = choices_list.mean(axis=0)
        quant_0 = np.quantile(regret_list, 0.25, axis=0)
        quant_1 = np.quantile(regret_list, 0.75, axis=0)
        meanDistance = distance_list.mean(axis=0)
        if choice:
            mean_it_choices = it_choices_list.mean(axis=0)

        print("meanreg_1 ", meanRegret)
        plt.figure("reg")
        plt.plot(1 + tsav, meanRegret,
                 color=colors[k], marker=styles[k], markevery=int(len(tsav) / 20), label=label
                 )
        plt.fill_between(1 + tsav, quant_0,
                         quant_1,
                         color=colors[k], alpha=0.2)

        plt.figure("mDis")
        plt.plot(1 + tsav, meanDistance, label=label, color=colors[k])
        # plt.xscale('log')

        plt.savefig("figures/" + "distance_" + name + ".pdf")

        if env.constraints.dim == 3:
            scatter3d(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
        if env.constraints.dim == 2:
            scatter_plot_2d_original_simplex(meanChoices, 1 + tsav, name="figures/" + "choices_" + name)
            if policy.iterate_bool and choice:
                scatter_2d_jet_reshaped_simplex(meanChoices, name="figures/" + "choices_" + name,
                                                name_policy=string_to_latex(policy.__class__.__name__),
                                                fun=env.function, contour=contour,
                                                iterates=mean_it_choices, split=split)

            else:
                scatter_2d_jet_reshaped_simplex(meanChoices, name="figures/" + "choices_" + name,
                                                name_policy=string_to_latex(policy.__class__.__name__),
                                                fun=env.function, contour=contour, split=split)
        if env.constraints.dim == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(tsav + 1, meanChoices[:])
            plt.savefig("figures/" + "choice" + name + ".pdf")
            plt.close()
        # plt.show()
    plt.figure("reg")
    # tick_font_size = 15
    # ax1.tick_params(labelsize=tick_font_size)
    plt.xlabel('Time')
    plt.ylabel('Regret')
    # plt.xlabel('Time', fontsize=10)
    # plt.ylabel('Regret', fontsize=10)
    if ylim_top:
        plt.ylim(top=ylim_top)
    plt.ylim(bottom=0)
    plt.legend(prop={'size': 10}, loc='upper left')
    # plt.legend(loc='upper left')
    plt.savefig("figures/comp_regret_" + name_pref + ".pdf")
    plt.figure("mDis")
    # plt.xscale('log')
    plt.legend(loc='upper left')
    # prop={'size': 10}
    # plt.tight_layout
    plt.savefig("figures/comp_distance_" + name_pref + ".pdf")

    plt.close("all")
