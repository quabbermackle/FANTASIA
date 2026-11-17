import numpy as np
from FANTASIA.dual import dual, dif, sin, cos

g3 = [f'g{a}{b}' for a, b in zip([0,1,2], [0,1,2])]
g4 = [f'g{a}{b}' for a, b in zip([0,1,2,3], [0,1,2,3])]

class metric4:
    
    def __init__(self, glist: dict):
        for key, val in glist.items():
            if key in g4:
                match key:
                    case 'g00': self.g00 = val
                    case 'g01': self.g01 = val
                    case 'g02': self.g02 = val
                    case 'g03': self.g03 = val

                    case 'g10': self.g10 = val
                    case 'g11': self.g11 = val
                    case 'g12': self.g12 = val
                    case 'g13': self.g13 = val

                    case 'g20': self.g20 = val
                    case 'g21': self.g21 = val
                    case 'g22': self.g22 = val
                    case 'g23': self.g23 = val

                    case 'g30': self.g30 = val
                    case 'g31': self.g31 = val
                    case 'g32': self.g32 = val
                    case 'g33': self.g33 = val

    # diagonal
    def g00(Param, Coord): return 0.
    def g11(Param, Coord): return 0.
    def g22(Param, Coord): return 0.
    def g33(Param, Coord): return 0.

    # off-diagonal (assume symmetric)
    def g01(Param, Coord): return 0.
    def g02(Param, Coord): return 0.
    def g03(Param, Coord): return 0.

    def g10(Param, Coord): return self.g01(Param, Coord)
    def g12(Param, Coord): return 0.
    def g13(Param, Coord): return 0.

    def g20(Param, Coord): return self.g02(Param, Coord)
    def g21(Param, Coord): return self.g12(Param, Coord)
    def g23(Param, Coord): return 0.

    def g30(Param, Coord): return self.g03(Param, Coord)
    def g31(Param, Coord): return self.g13(Param, Coord)
    def g32(Param, Coord): return self.g23(Param, Coord)
    

################### Kerr Metric Components ###################

# Diagonal components of the metric
def g00(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return -(r**2.+a**2.+2.*M*r*a**2.*sin(theta)**2./rhosq)/Delta
def g11(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return (a**2.-2.*M*r+r**2.)/(r**2.+a**2.*cos(theta)**2.)
def g22(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    return 1./(r**2.+a**2.*cos(theta)**2.)
def g33(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return (1./(Delta*sin(theta)**2.))*(1.-2.*M*r/rhosq)

# Off-diagonal components of the metric
def g01(Param,Coord):
    return 0
def g02(Param,Coord):
    return 0
def g03(Param,Coord):
    M = Param[0]
    a = Param[1]
    r = Coord[1]
    theta = Coord[2]
    Delta = r**2.-2.*M*r+a**2.
    rhosq = r**2.+a**2.*cos(theta)**2.
    return -(2.*M*r*a)/(rhosq*Delta)
def g12(Param,Coord):
    return 0
def g13(Param,Coord):
    return 0
def g23(Param,Coord):
    return 0

################### Metric Derivatives ###################

def metric_derivative(Param,Coord,metric,wrt):
    ''' wrt = 0,1,2,3 '''
    point_d = Coord[wrt]

    point_0 = dual(Coord[0],0)
    point_1 = dual(Coord[1],0)
    point_2 = dual(Coord[2],0)
    point_3 = dual(Coord[3],0)

    if metric == 'g00':
        if wrt == 0:
            return dif(lambda p:g00(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g00(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g00(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g00(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g11':
        if wrt == 0:
            return dif(lambda p:g11(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g11(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g11(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g11(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g22':
        if wrt == 0:
            return dif(lambda p:g22(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g22(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g22(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g22(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g33':
        if wrt == 0:
            return dif(lambda p:g33(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g33(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g33(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g33(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g44':
        if wrt == 0:
            return dif(lambda p:g44(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g44(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g44(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g44(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g01':
        if wrt == 0:
            return dif(lambda p:g01(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g01(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g01(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g01(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g02':
        if wrt == 0:
            return dif(lambda p:g02(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g02(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g02(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g02(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g03':
        if wrt == 0:
            return dif(lambda p:g03(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g03(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g03(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g03(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g12':
        if wrt == 0:
            return dif(lambda p:g12(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g12(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g12(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g12(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g13':
        if wrt == 0:
            return dif(lambda p:g13(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g13(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g13(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g13(Param,[point_0,point_1,point_2,p]),point_d)
    elif metric == 'g23':
        if wrt == 0:
            return dif(lambda p:g23(Param,[p,point_1,point_2,point_3]),point_d)
        elif wrt == 1:
            return dif(lambda p:g23(Param,[point_0,p,point_2,point_3]),point_d)
        elif wrt == 2:
            return dif(lambda p:g23(Param,[point_0,point_1,p,point_3]),point_d)
        elif wrt == 3:
            return dif(lambda p:g23(Param,[point_0,point_1,point_2,p]),point_d)
