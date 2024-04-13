import numpy as np


# Define the state transition function
def f(x, c):

    return np.array([x[0,0] + x[1,0] * np.cos(x[2,0]), x[1,0] * np.sin(x[2,0]), x[2,0] + c[0, 0], x[3,0]]).reshape(4,1)


# Define the measurement and state association function
def h(x):

    range = np.sqrt(x[0,0] ** 2 + x[1,0] ** 2)
    vel_norm = np.sqrt(x[2,0] ** 2 + x[3,0] ** 2)
    velocity = vel_norm * np.cos(np.arcsin(x[0,0] / range))
    AoA = np.arctan2(x[1,0], x[0,0])
    return np.array([range, velocity, AoA]).reshape(3,1)


# Define the EKF function
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

# THIS IS THE FIRST EKF APPLICATION
# Define the initial state estimate X0 (posX, posY, velX, velY)
X0 = np.array([-338.0000, -171.3000, 28.7406, 8.6010])  # initial radar measurements
X0 = X0.reshape((4, 1))

# Define the initial state covariance matrix P0 (initial uncertainty)
P0 = np.array([[0.5, 0, 0, 0],
               [0, 0.5, 0, 0],
               [0, 0, 0.5, 0],
               [0, 0, 0, 0.5]])

# Define the measurement noise covariance matrix R
R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])  # measurement standard deviations

# Define the system dynamics matrix u and the process noise covariance matrix Q
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])  # system dynamics
Q = np.diag([0.1 ** 2, 0.05 ** 2, 0.2 ** 2, 0.15 ** 2])  # process noise standard deviations

# load the dataset
data = np.loadtxt("outputRadar2.txt", delimiter=",")

# Loop over all measurements
for k in range(data.shape[0]): # k is a row (time, range, velocity, AoA)
    # Get the current measurement vector Z(k)
    Z_k = np.array([data[k, 1], data[k, 2], data[k, 3]]).reshape((3, 1))

    # Run the EKF algorithm
    X_k, P_k = EKF(X0, P0, Z_k, u, Q, R)

    # Store the new state estimate for the next time step
    X0 = X_k
    P0 = P_k

    # store the EKF results
    with open("EKF1_results.txt", "a") as l:
        np.savetxt(l, X_k.T, fmt='%.10f', delimiter=', ')
        #np.savetxt(f, np.hstack((X_k, np.diag(P_k).reshape(4,1))), delimiter=", ")




# THIS IS THE SECOND EKF APPLICATION
# Define the initial state estimate X0 (posX, posY, velX, velY)
X0 = np.array([-320.0000, -160.3000, 30.1206, 8.5127])  # initial radar measurements
X0 = X0.reshape((4, 1))

# Define the initial state covariance matrix P0 (initial uncertainty)
P0 = np.array([[0.25, 0, 0, 0],
               [0, 0.25, 0, 0],
               [0, 0, 0.25, 0],
               [0, 0, 0, 0.25]])

# Define the measurement noise covariance matrix R
R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])  # measurement standard deviations

# Define the system dynamics matrix u and the process noise covariance matrix Q
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])  # system dynamics
Q = np.diag([0.2 ** 2, 0.5 ** 2, 0.025 ** 2, 0.3 ** 2])  # process noise standard deviations

# load the dataset
data = np.loadtxt("outputRadar2.txt", delimiter=",")

# Loop over all measurements
for k in range(data.shape[0]): # k is a row (time, range, velocity, AoA)
    # Get the current measurement vector Z(k)
    Z_k = np.array([data[k, 1], data[k, 2], data[k, 3]]).reshape((3, 1))

    # Run the EKF algorithm
    X_k, P_k = EKF(X0, P0, Z_k, u, Q, R)

    # Store the new state estimate for the next time step
    X0 = X_k
    P0 = P_k

    # store the EKF results
    with open("EKF2_results.txt", "a") as l:
        np.savetxt(l, X_k.T, fmt='%.10f', delimiter=', ')
        #np.savetxt(f, np.hstack((X_k, np.diag(P_k).reshape(4,1))), delimiter=", ")



# THIS IS THE THIRD EKF APPLICATION
# Define the initial state estimate X0 (posX, posY, velX, velY)
X0 = np.array([-380.0120, -190.3500, 25.9406, 8.3010])  # initial radar measurements
X0 = X0.reshape((4, 1))

# Define the initial state covariance matrix P0 (initial uncertainty)
P0 = np.array([[2, 0, 0, 0],
               [0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 2]])

# Define the measurement noise covariance matrix R
R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])  # measurement standard deviations

# Define the system dynamics matrix u and the process noise covariance matrix Q
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])  # system dynamics
Q = np.diag([0.15 ** 2, 0.2 ** 2, 0.2 ** 2, 0.1 ** 2])  # process noise standard deviations

# load the dataset
data = np.loadtxt("outputRadar2.txt", delimiter=",")

# Loop over all measurements
for k in range(data.shape[0]): # k is a row (time, range, velocity, AoA)
    # Get the current measurement vector Z(k)
    Z_k = np.array([data[k, 1], data[k, 2], data[k, 3]]).reshape((3, 1))

    # Run the EKF algorithm
    X_k, P_k = EKF(X0, P0, Z_k, u, Q, R)

    # Store the new state estimate for the next time step
    X0 = X_k
    P0 = P_k

    # store the EKF results
    with open("EKF3_results.txt", "a") as l:
        np.savetxt(l, X_k.T, fmt='%.10f', delimiter=', ')
        #np.savetxt(f, np.hstack((X_k, np.diag(P_k).reshape(4,1))), delimiter=", ")




# THIS IS THE 4th EKF APPLICATION
# Define the initial state estimate X0 (posX, posY, velX, velY)
X0 = np.array([-315.0000, -163.3000, 26.7406, 8.2310])  # initial radar measurements
X0 = X0.reshape((4, 1))

# Define the initial state covariance matrix P0 (initial uncertainty)
P0 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

# Define the measurement noise covariance matrix R
R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])  # measurement standard deviations

# Define the system dynamics matrix u and the process noise covariance matrix Q
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])  # system dynamics
Q = np.diag([0.3 ** 2, 0.025 ** 2, 0.15 ** 2, 0.2 ** 2])  # process noise standard deviations

# load the dataset
data = np.loadtxt("outputRadar2.txt", delimiter=",")

# Loop over all measurements
for k in range(data.shape[0]): # k is a row (time, range, velocity, AoA)
    # Get the current measurement vector Z(k)
    Z_k = np.array([data[k, 1], data[k, 2], data[k, 3]]).reshape((3, 1))

    # Run the EKF algorithm
    X_k, P_k = EKF(X0, P0, Z_k, u, Q, R)

    # Store the new state estimate for the next time step
    X0 = X_k
    P0 = P_k

    # store the EKF results
    with open("EKF4_results.txt", "a") as l:
        np.savetxt(l, X_k.T, fmt='%.10f', delimiter=', ')
        #np.savetxt(f, np.hstack((X_k, np.diag(P_k).reshape(4,1))), delimiter=", ")




# THIS IS THE 5th EKF APPLICATION
# Define the initial state estimate X0 (posX, posY, velX, velY)
X0 = np.array([-342.0000, -175.3000, 28.7896, 8.6500])  # initial radar measurements
X0 = X0.reshape((4, 1))

# Define the initial state covariance matrix P0 (initial uncertainty)
P0 = np.array([[0.5, 0, 0, 0],
               [0, 0.5, 0, 0],
               [0, 0, 0.5, 0],
               [0, 0, 0, 0.5]])

# Define the measurement noise covariance matrix R
R = np.diag([114.5092776567171 ** 2, 8.867235477538927 ** 2, 1.4504739853497381 ** 2])  # measurement standard deviations

# Define the system dynamics matrix u and the process noise covariance matrix Q
u = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])  # system dynamics
Q = np.diag([0.15 ** 2, 0.05 ** 2, 0.15 ** 2, 0.1 ** 2])  # process noise standard deviations

# load the dataset
data = np.loadtxt("outputRadar2.txt", delimiter=",")

# Loop over all measurements
for k in range(data.shape[0]): # k is a row (time, range, velocity, AoA)
    # Get the current measurement vector Z(k)
    Z_k = np.array([data[k, 1], data[k, 2], data[k, 3]]).reshape((3, 1))

    # Run the EKF algorithm
    X_k, P_k = EKF(X0, P0, Z_k, u, Q, R)

    # Store the new state estimate for the next time step
    X0 = X_k
    P0 = P_k

    # store the EKF results
    with open("EKF5_results.txt", "a") as l:
        np.savetxt(l, X_k.T, fmt='%.10f', delimiter=', ')
        #np.savetxt(f, np.hstack((X_k, np.diag(P_k).reshape(4,1))), delimiter=", ")