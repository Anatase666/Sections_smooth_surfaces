from vpython import *
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def newtsolve(f, df, x0, tol=1e-6, max_iter=100):
    """
    Решает нелинейное уравнение f(x) = 0
    с помощью метода Ньютона.
    f - функция, для которой ищется корень
    df - производная функции f
    x0 - начальное приближение
    tol - допустимая погрешность
    max_iter - максимальное число итераций

    Возвращает корень уравнения и число итераций.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i
        dfx = df(x)
        if dfx == 0:
            break
        x = x - fx / dfx
    return None, i


class Line:
    def __init__(self, direct=None, spot=None):
        if direct is None:
            direct = vector(0, 0, 0)
        if spot is None:
            spot = vector(0, 0, 0)
        self.direct = norm(direct)
        self.spot = spot


class Plane:
    def __init__(self, line=None, angle=None, normal=None, spot=None):

        if normal is not None and spot is not None:
            self.spot = spot
            self.normal = normal
            self.angle = diff_angle(normal, vector(0, 0, 1))
            self.line = Line()
            self.line.spot = spot
            self.line.direct = norm(cross(normal, vector(0, 0, 1)))

        elif line is not None and angle is not None:
            self.line = line
            self.spot = vector(0, 0, 0)
            self.spot = line.spot
            self.normal = norm(rotate(vector(0, 0, 1), angle=angle, axis=line.direct))
        else:
            raise TypeError('Plane is not initialised')

        self.direct_vec1 = rotate(self.normal, angle=pi/2, axis=self.line.direct)
        self.direct_vec2 = norm(cross(vector(0, 0, 1), self.normal))
        # two complanar vectors#

    def calc_coord_from_param(self, p1, p2):
        # calculate coordinates of plane's spots using two params
        x1 = self.spot.x + p1 * self.direct_vec1.x + p2 * self.direct_vec2.x
        y1 = self.spot.x + p1 * self.direct_vec1.y + p2 * self.direct_vec2.y
        z1 = self.spot.x + p1 * self.direct_vec1.z + p2 * self.direct_vec2.z
        return x1, y1, z1

    def calc_borders_for_params(self, xmin, ymin, xmax, ymax):
        b_a = self.direct_vec2.y/self.direct_vec2.x
        m_min = (ymin - self.spot.x - b_a*(xmin - self.spot.x))/(self.direct_vec1.y - b_a*self.direct_vec1.x)
        n_min = (xmin - self.spot.x - self.direct_vec1.x*m_min)/self.direct_vec2.x

        m_max = (ymax - self.spot.x - b_a * (xmax - self.spot.x)) / (self.direct_vec1.y - b_a * self.direct_vec1.x)
        n_max = (xmax - self.spot.x - self.direct_vec1.x * m_max) / self.direct_vec2.x
        return m_min, n_min, m_max, n_max


    def calc_equation_of_plane(self):
        # gets symbolic system of equations for plane
        m, n = sp.symbols('m n')
        x1 = self.spot.x + m * self.direct_vec1.x + n * self.direct_vec2.x
        y1 = self.spot.x + m * self.direct_vec1.y + n * self.direct_vec2.y
        z1 = self.spot.x + m * self.direct_vec1.z + n * self.direct_vec2.z
        return x1, y1, z1

    def draw(self, xmin, ymin, xmax, ymax):
        vert1 = vertex(pos=self.spot)
        vert2 = vertex(pos=self.spot+sqrt((xmax-xmin)*(ymax-ymin))*self.direct_vec1)
        vert3 = vertex(pos=self.spot+sqrt((xmax-xmin)*(ymax-ymin))*self.direct_vec2)
        vert4 = vertex(pos=self.spot-sqrt((xmax-xmin)*(ymax-ymin))*self.direct_vec1)
        vert5 = vertex(pos=self.spot-sqrt((xmax-xmin)*(ymax-ymin))*self.direct_vec2)

        triangle(vs=[vert1, vert2, vert3], color=vector(1, 0, 0))
        triangle(vs=[vert1, vert3, vert4], color=vector(1, 0, 0))
        triangle(vs=[vert1, vert4, vert5], color=vector(1, 0, 0))
        triangle(vs=[vert1, vert5, vert2], color=vector(1, 0, 0))


def rangein(x1, x2, dx):
    # makes a list of values with equal gaps
    L = []
    L.append(x1)
    x = x1
    while(x < x2):
        x += dx
        L.append(x)
    return L


def FindIn2D(LV, dot, dx):
    # finds discret coordinates of spot in 2D list
    length = len(LV)
    for nx in range(length):
        for ny in range(length):
            if abs(nx.pos.x-dot[0])<=dx and abs(ny.pos.y-dot[1]) <= dx:
                return nx, ny


def model_func(function, xmin, ymin, xmax, ymax, dx):

    def funcProf(x1, y1):
        # determines a 3D profile
        f = function.evalf(subs={x: x1, y: y1})
        return f
    LV= []
    zmax = -1e100
    zmin = 1e100

    for x1 in rangein(xmin, xmax, dx):
        # making list of vertices
        TempLV = []
        for y1 in rangein(ymin, ymax, dx):
            z1 = funcProf(x1, y1)
            zmax = max(zmax, z1)
            zmin = min(zmin, z1)
            TempLV.append(vertex(pos=vector(x1, y1, z1), color=vector(1, 0.1, 1)))
        LV.append(TempLV)
        N = len(LV)
        for ix in range(N):
            # setting gradient coloring
            for iy in range(N):
                blue = LV[ix][iy].pos.z/zmax
                if blue > 0:
                    green = 1-blue
                    red = 0
                if blue < 0:
                    green = blue
                    blue = 1 - blue
                    red = 0
                LV[ix][iy].color = vector(red, green, blue)
        for ix in range(1, N-1):
            # drawing smooth surface
            for iy in range(1, N-1):
                triangle(vs=[LV[ix][iy], LV[ix+1][iy], LV[ix][iy+1]])
                triangle(vs=[LV[ix][iy], LV[ix - 1][iy], LV[ix][iy + 1]])
                triangle(vs=[LV[ix][iy], LV[ix + 1][iy], LV[ix][iy - 1]])
                triangle(vs=[LV[ix][iy], LV[ix - 1][iy], LV[ix][iy - 1]])
    return zmin, zmax, LV


dx = 0.1
# differential step
xmin = -1
ymin = -1
xmax = 1
ymax = 1
# borders of of being analysed function
x, y, z = sp.symbols("x y z")
function = sp.E**(-(x**2+y**2-x**3*y+sp.sin(x)))
# setting function with 2 variables
zmin, zmax, LV = model_func(function, xmin, ymin, xmax, ymax, dx)
line1 = Line(vector(1, 1, 0), vector(0, 0, 0.3))
plane1 = Plane(normal=vector(1, 1, 0), spot=vector(0, 0, 0))
plane1.draw(xmin, ymin, xmax, ymax)
# drawing a plane
x_symb, y_symb, z_symb = plane1.calc_equation_of_plane()
# substitution expressions for x, y, z
sect_equation = function.subs(x, x_symb)
sect_equation = sect_equation.subs(y, y_symb)
sect_equation = sect_equation - z
# section equation with 2 variables is ready for solving
print(sect_equation)

m_min, n_min, m_max, n_max = plane1.calc_borders_for_params(xmin, ymin, xmax, ymax)
if m_min > m_max: m_min, m_max = m_max, m_min
if n_min > n_max: n_min, n_max = n_max, n_min
for m in rangein(m_min, m_max, dx):
    n = newtsolve