import numpy as np
import unittest
from kf import KF





class TestKF(unittest.TestCase):
    def test_init(self):
        x = 2
        xv = 2
        y = 2
        yv = 2
        
        kf = KF(x,xv,y,yv, acceleration_variance=0.2)
        self.assertEqual(kf.position[0], x)
        self.assertEqual(kf.velocity[0], xv)
        self.assertEqual(kf.position[1], y)
        self.assertEqual(kf.velocity[1], yv)
        
        
    def test_after_prediction(self):
        x = 2
        xv = 2
        y = 2
        yv = 2
        
        kf = KF(x,xv,y,yv, acceleration_variance=1.2)
        kf.prediction(0.1)
        self.assertEqual(kf.covariance.shape, (4,4))
        self.assertEqual(kf.mean.shape, (4,))
        
    def test_after_prediction_increases_state_uncertainty(self):
        x = 2
        xv = 2
        y = 2
        yv = 2
        
        kf = KF(x,xv,y,yv, acceleration_variance=1.2)
        
        for i in range(10):
            det_before = np.linalg.det(kf.covariance)
            kf.prediction(0.1)
            det_after = np.linalg.det(kf.covariance)

        self.assertGreater(det_after, det_before)   
        
    
    def test_calling_update_does_not_crash(self):
        x = 2
        xv = 2
        y = 2
        yv = 2
        
        kf = KF(x,xv,y,yv, acceleration_variance=1.2)
        kf.update(0.1, 0.1)
    
    
    def test_calling_update_decrease_state_uncertainty(self):
        x = 2
        xv = 2
        y = 2
        yv = 2
        
        kf = KF(x,xv,y,yv, acceleration_variance=1.2)
        det_before = np.linalg.det(kf.covariance)
        kf.update(0.1, 0.01)
        det_after = np.linalg.det(kf.covariance)
        
        self.assertLess(det_after, det_before)
        
        