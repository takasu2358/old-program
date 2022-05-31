import math
import numpy as np
import robotmath as rm
import warnings
warnings.simplefilter('ignore')

def jacobian(nxtrobot, armid="rgt"):
    """
    compute the jacobian matrix of rgt or lft arm

    :param nxtrobot: see the nextage.NxtRobot class
    :param armid: a string indicating "rgt" or "lft"
    :return: armjac a 6-by-6 ndarray

    author: weiwei
    date: 20161111
    """

    if armid!="rgt" and armid!="lft":
        raise ValueError

    armlj = nxtrobot.rgtarm
    if armid == "lft":
        armlj = nxtrobot.lftarm

    armjac = np.zeros((6,6))
    for i in range(6):
        a = np.dot(armlj[i+1]["rotmat"], armlj[i+1]["rotax"])
        armjac[:, i] = np.append(np.cross(a, armlj[6]["linkpos"]-armlj[i+1]["linkpos"]), a)

    return armjac

def tcperror(nxtrobot, tgtpos, tgtrot, armid="rgt"):
    """
    compute the error of a specified (rgt or lft) tool point center to its goal

    :param nxtrobot: see the nextage.NxtRobot class
    :param armid: a string indicating "rgt" or "lft"
    :param tgtpos: the position of the goal
    :param tgtrot: the rotation of the goal
    :return: a 1-by-6 vector where the first three indicates the displacement in pos,
                the second three indictes the displacement in rot

    author: weiwei
    date: 20161111
    """

    if armid!="rgt" and armid!="lft":
        raise ValueError

    armlj = nxtrobot.rgtarm
    if armid == "lft":
        armlj = nxtrobot.lftarm

    deltapos = tgtpos - armlj[6]["linkend"]
    deltarot = np.dot(tgtrot, armlj[6]["rotmat"].transpose())

    anglesum = np.trace(deltarot)
    if anglesum == 3:
        deltaw = np.array([0,0,0])
    else:
        # revised on 20161216 at sapporo
        # compute the geodesic distance of two rotationmatrix
        # logarithm of R'R
        # see pygeometry.geodesic_distance_for_rotations for details
        nominator = anglesum-1
        if nominator > 2:
            nominator = 2
        if nominator < -2:
            nominator = -2
        theta = math.acos(nominator/2.0)
        if theta == 0:
            deltaw = np.array([0,0,0])
        else:
            deltaw = (theta/(2*math.sin(theta)))*(np.array([deltarot[2,1]-deltarot[1,2], \
                                                            deltarot[0,2]-deltarot[2,0], \
                                                            deltarot[1,0]-deltarot[0,1]]))

    return np.append(deltapos, deltaw)

def numik(nxtrobot, tgtpos, tgtrot, armid="lft"):
    """
    solve the ik numerically for the specified armid

    :param nxtrobot: see nextage.NxtRobot class
    :param tgtpos: the position of the goal, 1-by-3 numpy ndarray  目標位置(１×３の行列)
    :param tgtrot: the orientation of the goal, 3-by-3 numpyndarray  目標角度(３×３の行列)
    :param armid: a string "rgt" or "lft" indicating the arm that will be solved  左右のアームどちらか(rgtなら右、lftなら左)
    :return: armjnts: a 1-by-6 numpy ndarray  出力はアームの各関節角度(１×６行列)

    author: weiwei
    date: 20161111
    """

    if armid!="rgt" and armid!="lft":
        raise ValueError

    # armlj = nxtrobot.rgtarm
    # if armid == "lft":
    #     armlj = nxtrobot.lftarm

    # stablizer
    steplength = 5
    steplengthinc = 10
    armjntssave = nxtrobot.getarmjnts(armid)
    armjntsiter = armjntssave.copy()
    errnormlast = 0.0
    nlocalencountered = 0
    for i in range(100):
        armjac = jacobian(nxtrobot, armid)
        if np.linalg.matrix_rank(armjac) == 6:
            err = tcperror(nxtrobot, tgtpos, tgtrot, armid)
            # print(armjac, err)
            dq = steplength * (np.linalg.lstsq(armjac, err))[0]
        else:
            print("The Jacobian Matrix of the specified arm is at singularity")
            break
        # print np.linalg.norm(err)
        errnorm = np.linalg.norm(err)
        if errnorm < 1:
            # print 'goal reached', armjntsiter
            # print "number of iteration ", i
            armjntsreturn = nxtrobot.getarmjnts(armid)
            nxtrobot.movearmfk(armjntssave, armid)
            return armjntsreturn
        else:
            # todo dq definition
            # judge local minima
            if abs(errnorm - errnormlast) < 1e-3:
                nlocalencountered += 1
                # print "local minima at iteration", i
                # print "n local encountered", nlocalencountered
                steplength = 3
                steplengthinc = 7
                if nlocalencountered > 2:
                    break
            else:
                if steplength < 50:
                    steplength = steplength + steplengthinc
            armjntsiter += dq
            armjntsiter = rm.cvtRngPM180(armjntsiter)
            bdragged, jntangles = nxtrobot.chkrngdrag(armjntsiter, armid)
            armjntsiter[:] = jntangles[:]
            nxtrobot.movearmfk(jntangles, armid)
            # print jntangles
            # import nxtplot
            # nxtplot.plotstick(base.render, nxtrobot)
        errnormlast = errnorm
        # print errnorm
    # print "out of max iteration"
    nxtrobot.movearmfk(armjntssave, armid)
    return None