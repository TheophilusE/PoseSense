import numpy as np

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps: return v
    return v / n

def quat_identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

def quat_from_two_vectors(a, b, eps=1e-8):
    a = normalize(a); b = normalize(b)
    c = np.cross(a, b)
    d = np.dot(a, b)
    if d < -1 + eps:
        # 180-degree: rotate around any orthogonal axis
        axis = normalize(np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0]))
        axis = normalize(np.cross(a, axis))
        return quat_from_axis_angle(axis, np.pi)
    s = np.sqrt((1+d)*2.0)
    invs = 1.0 / s
    return normalize(np.array([s*0.5, c[0]*invs, c[1]*invs, c[2]*invs]))

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def quat_rotate(q, v):
    # rotate vector v by quaternion q
    qv = np.array([0, *v], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[1:]

def swing_twist_decomposition(q, twist_axis):
    # Project rotation onto twist axis
    twist_axis = normalize(twist_axis)
    # Rotate basis axis by q, compute twist
    r = q.copy()
    # get the vector part along the axis
    v = r[1:]
    proj = twist_axis * np.dot(v, twist_axis)
    twist = normalize(np.array([r[0], *proj]))
    swing = quat_mul(r, quat_conjugate(twist))
    return swing, twist

class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.02, dcutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / max(self.freq, 1e-6)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        ad = self._alpha(self.dcutoff)
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat
