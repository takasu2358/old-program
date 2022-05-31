import numpy as np
import math

def rodrigues(axis, theta):
    """
    Compute the rodrigues matrix using the given axis and theta

    ## input
    axis:
        a 1-by-3 numpy array list
    theta:
        angle in degree
    mat:
        a 3-by-3 numpy array, rotation matrix if this was not given, users could get it from return

    ## output
    the mat

    author: weiwei
    date: 20161220
    """

    theta = theta*math.pi/180.0
    axis = np.array([axis[0], axis[1], axis[2]])
    axis = axis/math.sqrt(np.dot(axis, axis))
    if theta > 2*math.pi:
        theta = theta % 2*math.pi
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2.0*(bc+ad), 2.0*(bd-ac)],
                     [2.0*(bc-ad), aa+cc-bb-dd, 2.0*(cd+ab)],
                     [2.0*(bd+ac), 2.0*(cd-ab), aa+dd-bb-cc]])

def cvtRngPM180(armjnts):
    """
    change the range of armjnts to +-180

    :param armjnts a numpyarray of jnts
    date: 20170330
    author: weiwei
    """

    armjntsnew = armjnts.copy()
    for i in range(armjntsnew.shape[0]):
        if armjntsnew[i] < 0:
            armjntsnew[i] = armjntsnew[i] % -360
            if armjntsnew[i] < -180:
                armjntsnew[i] = armjntsnew[i] + 360
        if armjnts[i] > 0:
            armjntsnew[i] = armjntsnew[i] % 360
            if armjntsnew[i] > 180:
                armjntsnew[i] = armjntsnew[i] - 360

    return armjntsnew