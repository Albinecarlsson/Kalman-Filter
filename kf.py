import numpy as np 


class KF:
    def __init__(self, initial_x: float, 
                 initial_x_vel: float,
                 initial_y: float, 
                 initial_y_vel: float, 
                 acceleration_variance: float) -> None:
        
        # mean of state GRV
        self._x = np.array([initial_x, initial_x_vel, initial_y, initial_y_vel])
        self._acceleration_variance = acceleration_variance
        # covariance of state GRV #usually we want to set this to a specific value
        self._P = np.eye(4)
        
    # dt is difference in time since last prediction
    def prediction(self, dt):
        F = np.array(   [[1, dt, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, dt],
                        [0, 0, 0, 1]]) 
        new_x = F.dot(self._x)
        
        G = np.array([[dt**2/2, 0],
                     [dt,      0],
                     [0, dt**2/2],
                     [0,      dt]])    
                 
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._acceleration_variance
       
        self._P = new_P
        self._x = new_x
        
    def update(self, measurement_value: float, measurement_variance: float):
        
        z = np.array(measurement_value)
        R = np.array(measurement_variance)
        
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        
        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        
        
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        
        new_x = self._x + K.dot(y)
        new_P = (np.eye(4) - K.dot(H)).dot(self._P)
        
        self._P = new_P
        self._x = new_x
        
    
    
    @property
    def covariance(self):
        return self._P
    
    @property
    def mean(self):
        return self._x 
        
    @property
    def position(self):
        return self._x[0:2]
    @property
    def velocity(self):
        return self._x[1:3]
    
    
        
