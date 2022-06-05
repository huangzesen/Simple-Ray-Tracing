import numpy as np
from scipy import constants
from scipy.integrate import RK45
mu_0 = constants.mu_0
epsilon_0 = constants.epsilon_0
m_e = constants.m_e
m_p = constants.m_p
e = constants.e

def n(x,y):
    """ calculate number density """
    n0 = 1e3 * 1e6
    # nv = n0 * (x**2+y**2)**(-1)
    nv = n0
    return nv

def Bvec(x,y):
    """ calculate magnetic field """
    # B0 = 3.12e-5
    # r = np.sqrt(x**2+y**2)
    # lamb = np.arctan(y/x)
    # Br = -2 * B0/(r**3) * np.sin(lamb)
    # Bt = B0/r**3 * np.cos(lamb)
    # Bn = 0
    # Bx = Br * np.cos(lamb) - Bt * np.sin(lamb)
    # By = Br * np.sin(lamb) + Bt * np.cos(lamb)
    # Bz = Bn
    By = 0
    Bx = 500 * 1e-9 + 10000000*x * 1e-9
    Bz = 0
    return Bx,By,Bz

def Bmod(x,y):
    """ calculate the modulus of magnetic field """
    Bx, By, Bz = Bvec(x,y)
    return np.sqrt(Bx**2+By**2+Bz**2)

def omega_p(x,y):
    """ Calculate omega p """
    wp = np.sqrt(
        n(x,y) * e**2/(m_e * epsilon_0)
    )
    return wp

def Omega_m(x,y):
    Om = e*Bmod(x,y)/m_e
    return Om

def Omega_p(x,y):
    Op = e*Bmod(x,y)/m_p
    return Op

def Lambda(x, y, theta, omega):
    L = omega/Omega_m(x,y) * (
        1 - (Omega_m(x,y)*Omega_p(x,y))/omega**2 + (Omega_m(x,y)/omega_p(x,y))**2
    )
    return L
    # return 0

def PartA(x, y, theta, omega):
    pa = omega_p(x,y)**2/(omega * Omega_m(x,y))
    return pa

def PartB(x, y, theta, omega):
    pb = 1./(np.cos(theta)-Lambda(x,y, theta,omega))
    return pb

def dpa_dx(x, y, theta, omega, dhx):
    dpadx = (PartA(x+dhx, y, theta, omega) - PartA(x, y, theta, omega))/dhx
    return dpadx

def dpa_dy(x, y, theta, omega, dhy):
    dpady = (PartA(x, y+dhy, theta, omega) - PartA(x, y, theta, omega))/dhy
    return dpady

def dpb_dx(x, y, theta, omega, dhx):
    dpbdx = (PartB(x+dhx, y, theta, omega) - PartB(x, y, theta, omega))/dhx
    return dpbdx
    
def dpb_dy(x, y, theta, omega, dhy):
    dpbdy = (PartB(x, y+dhy, theta, omega) - PartB(x, y, theta, omega))/dhy
    return dpbdy

# def dpb_dtheta(x,y,theta,omega,dhtheta):
#     dpbdtheta = (PartB(x, y, theta+dhtheta, omega) - PartB(x, y, theta, omega))/dhtheta
#     return dpbdtheta


class SimpleRayTracing:

    def __init__(self):
        self.variables = {
            'omega': 10**4,
            'dhx': 1e-6,
            'dhy': 1e-6,
            'dhchi': 1e-6
        }

        self.inity = np.array([
            5.0,0.0,np.pi/2*10/9
        ])

        self.dxdts = []
        self.dydts = []
        self.dchidts = []

    def rk45_init(self):
        self.rk45 = RK45(self.dylist_dt, 0, self.inity, 100, max_step = 0.001)

    def calc_theta(self,x,y,chi):
        v1 = np.array([np.cos(chi), np.sin(chi),0])
        v2 = Bvec(x,y)/Bmod(x,y)
        self.v1 = v1
        self.v2 = v2
        theta = np.arccos(np.sum(v1*v2))
        if theta > np.pi/2:
            theta = np.pi - theta
        self.theta = theta
        # ang2 = np.arctan((Bvec(x,y)/Bmod(x,y))[1]/(Bvec(x,y)/Bmod(x,y))[0])
        # theta1 = chi - ang2
        # self.theta1 = theta1
        return theta

    def dtheta_dchi(self,x,y,chi, dhchi):
        self.dtheta_dchi_val = (self.calc_theta(x,y,chi+dhchi) - self.calc_theta(x,y,chi))/dhchi

        return (self.calc_theta(x,y,chi+dhchi) - self.calc_theta(x,y,chi))/dhchi

    def dylist_dt(self, t ,ylist):
        """ ylist = [x,y,chi] """
        x,y,chi = ylist[0], ylist[1], ylist[2]

        theta = self.calc_theta(x,y,chi)
        self.theta = theta
        self.muval = self.mu(x,y,theta)
        
        dxdt = 1/self.mu(x,y,theta)**2 * (self.mu(x,y,theta) * np.cos(chi)+self.dmu_dchi(x,y,theta,chi)*np.sin(chi))
        # print(dxdt)
        self.dxdts.append(dxdt)
        self.lambda_val = Lambda(x,y,theta,self.variables['omega'])
        self.partb_val = PartB(x,y,theta,self.variables['omega'])

        dydt = 1/self.mu(x,y,theta)**2 * (self.mu(x,y,theta) * np.sin(chi)-self.dmu_dchi(x,y,theta,chi)*np.cos(chi))
        # print(dydt)
        self.dydts.append(dydt)

        dchidt = 1/self.mu(x,y,theta)**2 * (self.dmu_dy(x,y,theta) * np.cos(chi)-self.dmu_dx(x,y,theta)*np.sin(chi))
        # print(dthetadt)
        self.dchidts.append(dchidt)

        return np.array([dxdt,dydt,dchidt])

    def dmu_dx(self, x,y,theta):
        omega = self.variables['omega']
        dhx = self.variables['dhx']
        dmudx = 1/(2*self.mu(x,y,theta)) * (
            PartB(x,y,theta,omega) * dpa_dx(x,y,theta,omega,dhx) 
            + 
            PartA(x,y,theta,omega) * dpb_dx(x,y,theta,omega,dhx)
        )
        return dmudx

    def dmu_dy(self, x,y,theta):
        omega = self.variables['omega']
        dhy = self.variables['dhy']
        dmudy = 1/(2*self.mu(x,y,theta)) * (
            PartB(x,y,theta,omega) * dpa_dy(x,y,theta,omega,dhy) 
            + 
            PartA(x,y,theta,omega) * dpb_dy(x,y,theta,omega,dhy)
        )
        return dmudy

    def dmu_dchi(self, x,y,theta, chi):
        omega = self.variables['omega']
        dhchi = self.variables['dhchi']
        return 1./2 * self.mu(x,y,theta)*PartB(x,y,theta,omega) * np.sin(theta) * self.dtheta_dchi(x,y,chi, dhchi)

    def mu(self, x,y,theta):
        omega = self.variables['omega']
        mu_val = np.sqrt(PartA(x,y,theta,omega)*PartB(x,y,theta,omega))
        return mu_val


