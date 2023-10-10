from fenics import *
from ufl import nabla_div, nabla_grad
from vedo.dolfin import plot
#import matplotlib.pyplot as plt

# scaled variables
L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# create mesh and define function space
#Boxmesh(x0,y0,z0, x,y,z,nx,ny,nz) create tetrahegonal mesh between two points with number of cells
mesh_ = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh_, 'P', 1)

# define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol

# DirichletBC(FunctionSpace, value, boundary)
bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# define strain and stress
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# define vairational problem
u = TrialFunction(V)
d = u.geometric_dimension()  #space dimension
v = TestFunction(V)
f = Constant((0, 0, -rho*g))   # gravity
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds

# compute solution
u = Function(V)
solve(a == L, u, bc)

# plot solution
#plot(u, title='Displacement', mode='displacement')  #Implemtn name error

# plot stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # dviatoric stress
von_Mises = sqrt(3./2*inner(s,s))
V = FunctionSpace(mesh_, 'P', 1)
von_Mises = project(von_Mises, V)
#plot(von_Mises, title='Stress Intensity')

# compute magnitue of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)

# compute strain energy density
strain_energy_density = 0.5*inner(sigma(u), epsilon(u))
SED= project(strain_energy_density, V)

# plot(u, mode="displacements with arrows", text='displaced mesh', interactive=False)
# plot(SED, text='strain energy density', new=True, size=(900,400), pos=(1100,0), zoom=2)
plot(mesh_)