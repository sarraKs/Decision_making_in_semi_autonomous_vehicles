import numpy as np


def f(x, c):

    return np.array([x[0,0] + x[1,0] * np.cos(x[2,0]), x[1,0] * np.sin(x[2,0]), x[2,0] + c[0, 0], x[3,0]]).reshape(4,1)


def h(x):

    range = np.sqrt(x[0,0] ** 2 + x[1,0] ** 2)
    vel_norm = np.sqrt(x[2,0] ** 2 + x[3,0] ** 2)
    velocity = vel_norm * np.cos(np.arcsin(x[0,0] / range))
    AoA = np.arctan2(x[1,0], x[0,0])
    return np.array([range, velocity, AoA]).reshape(3,1)


# Jacobian of h
def H(x):
    range = np.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)
    vel_norm = np.sqrt(x[2, 0] ** 2 + x[3, 0] ** 2)
    return np.array([[x[0, 0] / range, x[1, 0] / range, 0, 0],
                  [-x[2, 0] * x[1, 0] / (range * vel_norm), x[2, 0] * x[0, 0] / (range * vel_norm), x[0, 0] / range,
                   x[1, 0] / range],
                  [-x[1, 0] / (x[0, 0] ** 2 + x[1, 0] ** 2), x[0, 0] / (x[0, 0] ** 2 + x[1, 0] ** 2), 0,
                   0]])

def EKF(x, P, z, c, Q, R):
    # Prediction step
    x_pred = f(x, c)
    F = np.array([[1, np.cos(x[2,0]), -x[1,0]*np.sin(x[2,0]),0], [0, np.sin(x[2,0]), x[1,0]*np.cos(x[2,0]),0], [0, 0, 1,0], [0,0,0,1]]) # Jacobian of f
    P_pred = F @ P @ F.T + Q
    # Update step
    range = np.sqrt(x[0,0] ** 2 + x[1,0] ** 2)
    vel_norm = np.sqrt(x[2,0] ** 2 + x[3,0] ** 2)
    H = np.array([[x[0,0]/range, x[1,0]/range, 0, 0], [-x[2,0] * x[1,0] / (range * vel_norm), x[2,0] * x[0,0] / (range * vel_norm), x[0,0]/range, x[1,0]/range], [-x[1,0] / (x[0,0]**2 + x[1,0]**2), x[0,0] / (x[0,0]**2 + x[1,0]**2),0,0]])  # Jacobian of h
    y = z - h(x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(4) - K @ H) @ P_pred
    return x_new, P_new


R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])
P1 = np.array([[0.5, 0, 0, 0],
               [0, 0.5, 0, 0],
               [0, 0, 0.5, 0],
               [0, 0, 0, 0.5]])
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
Q = np.diag([0.1 ** 2, 0.05 ** 2, 0.2 ** 2, 0.15 ** 2])
X1 = np.array([-338.0000, -171.3000, 28.7406, 8.6010]).reshape((4, 1))


Xdata = np.loadtxt("EKFagg_results.txt", delimiter=",")
Zdata = np.loadtxt("outputRadar2.txt", delimiter=",")

for k in range(Zdata.shape[0]):
    Z_k = np.array([Zdata[k, 1], Zdata[k, 2], Zdata[k, 3]]).reshape((3, 1))
    X_k = np.array([Xdata[k, 0], Xdata[k, 1], Xdata[k, 2], Xdata[k, 3]]).reshape((4, 1))

    Z_r = Z_k - h(X_k)  # define the measurement residual

    print(Z_r)

    XEKF, PEKF = EKF(X1, P1, Z_k, u, Q, R)
    X1 = XEKF
    P1 = PEKF

    P_k = H(X_k) @ PEKF @ H(X_k).T + R  # Update the process uncertainty

    U = np.sqrt(Z_r.T @ np.linalg.inv(P_k) @ Z_r)  # Mahalanobis distances (sensor uncertainty)

    print(U)

    with open("Uncertainties.txt", "a") as l:
        np.savetxt(l, U.T, fmt='%.10f', delimiter=', ')


