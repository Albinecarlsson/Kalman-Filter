from turtle import position
import numpy as np
import matplotlib.pyplot as plt

from kf import KF

plt.ion()
plt.figure()

kf = KF(0,1,0,1, acceleration_variance=0.2)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

real_pos = np.array([0.0, 0.0])
real_velocity = np.array([0.9, 0.9])
meas_variance = 0.1 ** 2

means = []
covs = []
real_xys = []
real_vxys = []

for i in range(NUM_STEPS):
    
    if i > 500:
        real_velocity = real_velocity * 0.99
    
    means.append(kf.mean)
    covs.append(kf.covariance)
    
    real_pos = real_pos + real_velocity.dot(DT)

    kf.prediction(DT)
    if i % MEAS_EVERY_STEPS == 0 and i != 0:
        kf.update(measurement_value=real_pos + np.random.randn() * np.sqrt(meas_variance), 
                  measurement_variance=meas_variance)
        
    real_xys.append(real_pos)
    real_vxys.append(real_velocity)



plt.subplot(2,2,1)
plt.title("position")
plt.plot([m[0] for m in means], label='x')
plt.plot([real_x[0] for real_x in real_xys], label='real x')


plt.plot([m[0] - 2*np.sqrt(covs[i][0,0]) for i,m in enumerate(means)],'r--' ,label='x - 2*sqrt(cov[0,0])')
plt.plot([m[0] + 2*np.sqrt(covs[i][0,0]) for i,m in enumerate(means)],'r--' ,label='x - 2*sqrt(cov[0,0])')


plt.subplot(2,2,2)
plt.title("position")
plt.plot([m[2] for m in means], label='y')
plt.plot([real_y[1] for real_y in real_xys], label='real y')

plt.plot([m[2] - 2*np.sqrt(covs[i][2,2]) for i,m in enumerate(means)],'r--', label='y - 2*sqrt(cov[2,2])')
plt.plot([m[2] + 2*np.sqrt(covs[i][2,2]) for i,m in enumerate(means)], 'r--',label='y + 2*sqrt(cov[2,2])')


plt.subplot(2,2,3)
plt.title("velocity")
plt.plot([m[1] for m in means], label='x')
plt.plot([real_vx[0] for real_vx in real_vxys], label='real x')

plt.plot([m[1] - 2*np.sqrt(covs[i][1,1]) for i,m in enumerate(means)], 'r--',label='x - 2*sqrt(cov[1,1])')
plt.plot([m[1] + 2*np.sqrt(covs[i][1,1]) for i,m in enumerate(means)], 'r--',label='x + 2*sqrt(cov[1,1])')

plt.subplot(2,2,4)
plt.title("velocity")
plt.plot([m[3] for m in means], label='y')
plt.plot([real_vx[1] for real_vx in real_vxys], label='real y')

plt.plot([m[3] - 2*np.sqrt(covs[i][3,3]) for i,m in enumerate(means)], 'r--',label='y - 2*sqrt(cov[3,3])')
plt.plot([m[3] + 2*np.sqrt(covs[i][3,3]) for i,m in enumerate(means)], 'r--',label='y + 2*sqrt(cov[3,3])')

plt.ginput(1)

