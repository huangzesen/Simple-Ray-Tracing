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
    return 1

def Bvec(x,y):
    """ calculate magnetic field """
    return 0, x, 0

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

def Lambda(x,y, theta,omega):
    L = omega/Omega_m(x,y) * (
        1 - (Omega_m(x,y)*Omega_p(x,y))/omega**2 + (Omega_m(x,y)/omega_p(x,y))**2
    )
    return L

def PartA(x,y, theta, omega):
    pa = omega_p(x,y)**2/(omega * Omega_m(x,y))
    return pa

def PartB(x,y, theta, omega):
    pb = 1./(np.cos(theta)-Lambda(x,y, omega))
    return pb

def dpa_dx(x,y, theta, omega, dhx):
    dpadx = (PartA(x+dhx, y, theta, omega) - PartA(x, y, theta, omega))/dhx
    return dpadx

def dpa_dy(x,y,theta,omega,dhy):
    dpady = (PartA(x, y+dhy, theta, omega) - PartA(x, y, theta, omega))/dhy
    return dpady

def dpb_dx(x,y,theta,omega,dhx):
    dpbdx = (PartB(x+dhx, y, theta, omega) - PartB(x, y, theta, omega))/dhx
    return dpbdx
    
def dpb_dy(x,y,theta,omega,dhy):
    dpbdy = (PartB(x, y+dhy, theta, omega) - PartB(x, y, theta, omega))/dhy
    return dpbdy

def dpb_dtheta(x,y,theta,omega,dhtheta):
    dpbdtheta = (PartB(x, y, theta+dhtheta, omega) - PartB(x, y, theta, omega))/dhtheta
    return dpbdtheta




class SimpleRayTracing:

    def __init__(self):
        self.variables = {
            'omega': 1,
            'dhx': 0.01,
            'dhy': 0.01,
            'dhtheta': 0.01
        }

        self.inity = np.array([
            0,0,0
        ])

    def rk45_init(self):
        self.rk45 = RK45(self.dylist_dt, 0, self.inity, 100)

    def dylist_dt(self, t ,ylist):
        """ ylist = [x,y,theta] """
        x,y,theta = ylist[0], ylist[1], ylist[2]
        
        dxdt = 1/self.mu(x,y,theta)**2 * (self.mu(x,y,theta) * np.cos(theta)+self.dmu_dtheta(x,y,theta)*np.sin(theta))

        dydt = 1/self.mu(x,y,theta)**2 * (self.mu(x,y,theta) * np.sin(theta)-self.dmu_dtheta(x,y,theta)*np.cos(theta))

        dthetadt = 1/self.mu(x,y,theta)**2 * (self.dmu_dy(x,y,theta) * np.cos(theta)-self.dmu_dx(x,y,theta)*np.sin(theta))

        return np.array([dxdt,dydt,dthetadt])

    def dmu_dx(self, x,y,theta):
        omega = self.variables['omega']
        dhx = self.variables['dhx']
        dmudx = 1/(2*self.mu(x,y,theta)) *(
            PartB(x,y,theta,omega) * dpa_dx(x,y,theta,omega,dhx) 
            + 
            PartA(x,y,theta,omega) * dpb_dx(x,y,theta,omega,dhx)
        )
        return dmudx

    def dmu_dy(self, x,y,theta):
        omega = self.variables['omega']
        dhy = self.variables['dhy']
        dmudy = 1/(2*self.mu(x,y,theta)) *(
            PartB(x,y,theta,omega) * dpa_dy(x,y,theta,omega,dhy) 
            + 
            PartA(x,y,theta,omega) * dpb_dy(x,y,theta,omega,dhy)
        )
        return dmudy

    def dmu_dtheta(self, x,y,theta):
        omega = self.variables['omega']
        return PartA(x,y,theta,omega) * PartB(x,y,theta,omega)**2 * np.sin(theta)

    def mu(self, x,y,theta):
        omega = self.variables['omega']
        mu_val = np.sqrt(PartA(x,y,theta,omega)*PartB(x,y,theta,omega))
        return mu_val


