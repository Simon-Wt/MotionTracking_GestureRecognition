import numpy as np


def getRotationMatrix(gravity, geomagnetic):
    """Computes a 3x3 rotation matrix transforming a vector
    from the device coordinate system to the world's coordinate system.
    gravity should be the last know gravity vector. Geomagnetic is the
    last know reading from the geomagnetic sensor.
    """
    Ax = gravity[0]
    Ay = gravity[1]
    Az = gravity[2]
    normsqA = (Ax * Ax + Ay * Ay + Az * Az)
    g = 9.81
    freeFallGravitySquared = 0.01 * g * g
    if normsqA < freeFallGravitySquared:
        # gravity less than 10% of normal value
        return None

    Ex = geomagnetic[0]
    Ey = geomagnetic[1]
    Ez = geomagnetic[2]
    Hx = Ey * Az - Ez * Ay
    Hy = Ez * Ax - Ex * Az
    Hz = Ex * Ay - Ey * Ax
    normH = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if normH < 0.1:
        # device is close to free fall (or in space?), or close to
        # magnetic north pole. Typical values are  > 100.
        return None

    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH
    Hz *= invH
    invA = 1.0 / np.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA
    Az *= invA
    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz
    Mz = Ax * Hy - Ay * Hx

    R = np.matrix([[Hx, Hy, Hz],
                   [Mx, My, Mz],
                   [Ax, Ay, Az]])

    # compute the inclination matrix by projecting the geomagnetic
    # vector onto the Z (gravity) and X (horizontal component
    # of geomagnetic vector) axes.
    #invE = 1.0 / np.sqrt(Ex * Ex + Ey * Ey + Ez * Ez);
    #c = (Ex * Mx + Ey * My + Ez * Mz) * invE;
    #s = (Ex * Ax + Ey * Ay + Ez * Az) * invE;
    # I = np.matrix([[ 1.0, 0.0, 0.0],
    #               [ 0.0, c, s ],
    #               [0.0, -s, -c]])

    return R
