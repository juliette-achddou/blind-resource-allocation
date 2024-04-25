# utils for the direct search methods

import numpy as np
from scipy.linalg import null_space


def gentech(A, Z, Ap, Am):
    """
    Using Theorem 4.1 of Griffin, Kolda, Lewis, 2008
    :param A:  matrix A where: l <=A.T X < U defines the constrained domain
    :param Z: unitary array of dimension d
    :param Ap: indices of the rows of A representing alpha binding constraints, where the constraints are
     violated from above
    :param Am: indices of the rows of A representing alpha binding constraints, where the constraints are
     violated from above
    :return:
    """
    # the alpha binding cone is defined by - V_P u + V_L xi where u>=0 
    # where  V_P, V_L are defined as in Griffin, Kolda and Lewis
    # V_P = -Z R, V_L = Z N where Z, R, N are described in Griffin, Kolda and Lewis

    n = Z.shape[0]
    Ib = [x for x in set([y for y in Ap]) & set([z for z in Am])]
    # violated  by above
    Iu = [x for x in set([y for y in Ap]) if x not in Ib]
    # violated  by below
    Il = [x for x in set([y for y in Am]) if x not in Ib]
    # both

    W = Z @ (Z.transpose() @ A.transpose())
    Vp = np.array([])
    if Iu:
        Vp = np.array([W[:, x] for x in Iu]).transpose()
    if Il:
        if Vp.size == 0:
            Vp = np.array([-W[:, x] for x in Il]).transpose()
        else:
            Vp = np.hstack((Vp, np.array([-W[:, x] for x in Il]).transpose()))
    Vl = []
    if Ib:
        Vl = np.array([-W[:, x] for x in Ib])
        Zl = null_space(Vl.transpose())
    else:
        Zl = Z
    if Vp.size == 0:
        V_L = Zl
        V_P = []
    elif Zl.size == 0:
        V_L = []
        V_P = []
    else:
        # Now we compute the generators
        Q = Zl.transpose() @ Vp
        rp = np.linalg.matrix_rank(Q)
        sp = Q.shape

        if rp == min(sp):
            # Non-degenerate case - Direct computation of the generators

            # if Q, R are the unitary matrix and the upper triangular matrix associated to the QR decomposition of
            # Q := Zl  Vp, where Vp is the matrix formed by the rows of the matrix A corresponding to violated
            # constraints (but only from above or below) then if we are in the non-degenerate case,  there exist
            # R s.t. V_P Z R  = I, and V_P = Z R and V_L = Z N where N is a matrix whose columns are a basis for
            # the nullspace of V T
            # We solve VP^T Z R = I then  R = Q1 ((R1)^T)^(-1) I
            # and Z R = Z Q1 ((R1)^^T)^(-1) I
            Qi, Ri = np.linalg.qr(Q, mode="complete")
            if sp[0] > sp[1]:
                # V_L = ZN
                V_L = Zl @ Qi[:, sp[1]:]
            else:
                V_L = []
            r2 = min(sp[1], Qi.shape[1])
            Qi = Qi[:, :r2]
            Ri = Ri[:r2, :r2]
            # Z R = I then  R = Q1 ((R1)^T)^(-1) I
            V_P = -Zl @ (Qi @ np.linalg.inv(Ri.transpose()))
        else:
            #		Degenerate case - calling the double description method
            V_L, V_P = [], []
            # TO DO
            calldd = 1
    return V_L, V_P


def activeconsAiZ(x, alpha, li, ui, A, Z):
    """
    Finds the alpha-binding constraints at x
    :param x: current iterate, at which one wants to find the alpha binding constraints (dimension d and not d+1)
    :param alpha: alpha
    :param li: vector li where: li <=A.T X < ui defines the constrained domain
    :param ui: vector ui where: li <=A.T X < ui defines the constrained domain
    :param A: matrix A where: li <=A.T X < ui defines the constrained domain
    :param Z: unitary array of dimension d
    :return: Iap : indices of the rows of A representing alpha binding constraints, where the constraints are
     violated from above, Iam : indices of the rows of A representing alpha binding constraints, where the constraints are
     violated from above, nbgen : number of constraints violated
    """
    nz = np.sqrt(np.diag(A @ Z @ Z.transpose() @ A.transpose()))
    iz = np.argwhere(nz < 1e-15)
    ni = A.shape[0]
    tolfeas = 1e-15
    Vp = np.inf * np.ones(ni)
    Vm = np.inf * np.ones(ni)
    for i in range(ni):
        Vp[i] = float((-(A[i, :] @ x - ui[i] - tolfeas)) / nz[i])
        Vm[i] = float(((A[i, :] @ x - li[i] + tolfeas)) / nz[i])
    if iz:
        for i in range(len(iz)):
            j = iz[i]
            if np.linalg.norm(A[j, :] * x - ui[j]) < tolfeas:
                Vp[j] = 0
            else:
                Vp[j] = np.inf
            if np.linalg.norm(A[j, :] @ x - li[j]) < tolfeas:
                Vm[j] = 0
            else:
                Vm[j] = np.inf

    Iap = np.argwhere(Vp <= alpha).flatten()
    Iam = np.argwhere(Vm <= alpha).flatten()
    nbgen = len(Iap) + len(Iam)
    return Iap, Iam, nbgen


def determine_set_dk(dim, A, l, u, x, alpha):
    """
    Constructs a set (almost minimal) D_k of directions positively generating the cone
     of alpha-binding constraints
    :param dim: dimension d
    :param A: matrix A where: li <=A.T X < ui defines the constrained domain
    :param l: vector l where: l <=A.T X < u defines the constrained domain
    :param u: vector u where: l <=A.T X < u defines the constrained domain
    :param x: current iterate, at which one wants to determine the set D_k of directions positively generating the cone
     of alpha-binding constraints
    :param alpha: radius
    :return:  set (almost minimal) D_k of directions positively generating the cone
     of alpha-binding constraints
    """
    Z = np.identity(dim)
    Iap, Iam, nbgen = activeconsAiZ(x, alpha, l, u, A, Z)
    Ys, Yc = gentech(A, Z, Iap, Iam)
    set_d_k = []
    if len(Ys) != 0:
        for ys in Ys.transpose():
            ys = ys / np.linalg.norm(ys)
            set_d_k.append(ys)
            set_d_k.append(-ys)
    if len(Yc) != 0:
        for yc in Yc.transpose():
            yc = yc / np.linalg.norm(yc)
            set_d_k.append(yc)
    return set_d_k
