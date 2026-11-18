import numpy as np
from matplotlib import pyplot as plt
import FANTASY

if False:
    # unstable spherical null geodesic
    # Kerr metric in Boyer-Lindquist coordinates
    # {t, r, theta, phi}

    # metric parameters
    M = 1. # mass
    a = 1. # spin
    Param = [M, a]

    # initial conditions
    q0 = [0., 1. + np.sqrt(3.), np.pi / 2., 0.] # position
    p0 = [-1., 0., np.sqrt(12. + 8. * np.sqrt(3.)), -1.] # momentum

    # integrator options
    N = 10000
    delta = 0.01
    omega = 5.0
    order = 2

if True:
    # timelike test particle
    # Kerr metric in Boyer-Lindquist coordinates
    # {t, r, theta, phi}

    # metric parameters
    M = 1. # mass
    a = 0.5 # spin
    Param = [M, a]

    # initial conditions
    q0 = [0., 20., np.pi / 2., 0.] # position
    p0 = [-0.9764550153430405, 0., 3.8, 3.] # momentum

    # integrator options
    # N = int(5e4)
    N = 1000
    delta = 0.5
    omega = 1.0
    order = 2

output = np.array(FANTASY.geodesic_integrator(N,delta,omega,q0,p0,Param,order))

q = output[:,0,:]
p = output[:,1,:]

cart = np.zeros_like(q)
cart[:,0] = q[:,0]
for i in range(q.shape[0]):
    cart[i,1] = FANTASY.CoordTrans1(Param, q[i,:])
    cart[i,2] = FANTASY.CoordTrans2(Param, q[i,:])
    cart[i,3] = FANTASY.CoordTrans3(Param, q[i,:])
    
fig1, ax1 = plt.subplots(1, subplot_kw={"projection": "3d"})

# ax1.plot(q[:,1], q[:,2], q[:,3])
ax1.plot(cart[:,1], cart[:,2], cart[:,3])

plt.show()
