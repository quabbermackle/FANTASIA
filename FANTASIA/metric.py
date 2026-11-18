import numpy as np
from FANTASIA.dual import dual, dif, sin, cos

class metric4:
    # Generic 4d metric interface.
    # The user must be aware whether child overrides are covariant or contravariant,
    # since the Hamiltonian for geodesics in curved space uses the contravariant metric.
    # If a reference defines a covariant metric, take the matrix inverse 
    # to find the contravariant form.

    def __init__(self, Param: dict):
        self.Param = Param

    # diagonal
    def g00(self, Coord): return 0.
    def g11(self, Coord): return 0.
    def g22(self, Coord): return 0.
    def g33(self, Coord): return 0.

    # off-diagonal
    def g01(self, Coord): return 0.
    def g02(self, Coord): return 0.
    def g03(self, Coord): return 0.

    def g12(self, Coord): return 0.
    def g13(self, Coord): return 0.

    def g23(self, Coord): return 0.

    # symmetric elements (DO NOT OVERRIDE)
    def g10(self, Coord): return self.g01(Coord)

    def g20(self, Coord): return self.g02(Coord)
    def g21(self, Coord): return self.g12(Coord)

    def g30(self, Coord): return self.g03(Coord)
    def g31(self, Coord): return self.g13(Coord)
    def g32(self, Coord): return self.g23(Coord)

################### Kerr Metric Components ###################

class Kerr_BL(metric4):
    # contravariant Kerr metric

    def __init(self, Param: dict):
        super.__init__(Param)

        self.M = Param['M'] if 'M' in Param.keys() else 1.0
        self.a = Param['a'] if 'a' in Param.keys() else 1.0

    def Delta(self, Coord):
        r = Coord[1]
        return r**2. - 2.*self.M*r + self.a**2.

    def rhosq(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        return r**2. + (self.a**2. * cos(theta)**2.)

    # Diagonal components of the metric
    def g00(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        Delta = self.Delta(Coord)
        rhosq = self.rhosq(Coord)
        return -(r**2. + self.a**2. + 2.*self.M*r*self.a**2.*sin(theta)**2. / rhosq) / Delta

    def g11(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        return (self.a**2. - 2.*self.M*r + r**2.) / (r**2. + self.a**2.*cos(theta)**2.)

    def g22(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        return 1. / (r**2. + self.a**2.*cos(theta)**2.)

    def g33(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        Delta = self.Delta(Coord)
        rhosq = self.rhosq(Coord)
        return (1. / (Delta*sin(theta)**2.)) * (1. - 2.*self.M*r / rhosq)

    # Off-diagonal components of the metric
    def g03(self, Coord):
        r = Coord[1]
        theta = Coord[2]
        Delta = self.Delta(Coord)
        rhosq = self.rhosq(Coord)
        return -(2.*self.M*r*self.a) / (rhosq*Delta)

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
