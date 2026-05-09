import numpy as np


class KF:
    """
    Kalman Filter for 2D position and velocity estimation.

    State vector: [x, vx, y, vy]
    Measurements: [x, y] (position only)
    """

    def __init__(self, initial_x: float,
                 initial_x_vel: float,
                 initial_y: float,
                 initial_y_vel: float,
                 acceleration_variance: float) -> None:
        """
        Initialize the Kalman Filter.

        Args:
            initial_x: Initial x position
            initial_x_vel: Initial x velocity
            initial_y: Initial y position
            initial_y_vel: Initial y velocity
            acceleration_variance: Process noise variance for acceleration
        """
        # mean of state GRV
        self._x = np.array([initial_x, initial_x_vel, initial_y, initial_y_vel], dtype=float)
        self._acceleration_variance = acceleration_variance

        # covariance of state GRV
        self._P = np.eye(4, dtype=float)

    def prediction(self, dt: float) -> None:
        """
        Predict the next state using the motion model.

        Args:
            dt: Time step (difference in time since last prediction)
        """
        # State transition matrix (constant velocity model)
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]], dtype=float)

        # Process noise matrix (relates acceleration to state change)
        G = np.array([[dt**2/2, 0],
                      [dt,      0],
                      [0, dt**2/2],
                      [0,      dt]], dtype=float)

        # Update state estimate
        new_x = F @ self._x

        # Update covariance estimate (process noise added)
        new_P = F @ self._P @ F.T + G @ G.T * self._acceleration_variance

        self._P = new_P
        self._x = new_x

    def update(self, measurement_value, measurement_variance: float) -> None:
        """
        Update state estimate with a measurement.

        Args:
            measurement_value: Measured position [x, y] or array-like
            measurement_variance: Measurement noise variance (scalar for both x and y)
        """
        # Convert measurement to array
        z = np.asarray(measurement_value, dtype=float).reshape(-1)
        if z.shape != (2,):
            raise ValueError("measurement_value must contain exactly two elements: [x, y]")

        # Measurement noise covariance (diagonal matrix)
        R = np.eye(2) * measurement_variance

        # Measurement matrix (we measure position only, not velocity)
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]], dtype=float)

        # Innovation (measurement residual)
        y = z - H @ self._x

        # Innovation covariance
        S = H @ self._P @ H.T + R
        PHt = self._P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T

        # Update state estimate
        self._x = self._x + K @ y

        # Update covariance estimate
        I = np.eye(4)
        I_KH = I - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

    @property
    def mean(self) -> np.ndarray:
        """Mean of the state estimate [x, vx, y, vy]."""
        return self._x

    @property
    def position(self) -> np.ndarray:
        """Estimated position [x, y]."""
        return self._x[[0, 2]]

    @property
    def velocity(self) -> np.ndarray:
        """Estimated velocity [vx, vy]."""
        return self._x[1:4:2]


    @property
    def covariance(self) -> np.ndarray:
        return self._P
