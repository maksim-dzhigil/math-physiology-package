import fenics as fe


mesh = fe.UnitCubeMesh(16, 16, 16)
V = fe.FunctionSpace(mesh, 'P', 1)
u_D_left = fe.Constant(0.)
u_D_right = fe.Constant(1.)


def boundary_left(x):
    return x[0] == 0

def boundary_right(x):
    return x[0] == 1


bc_left = fe.DirichletBC(V, u_D_left, boundary_left)
bc_right = fe.DirichletBC(V, u_D_right, boundary_right)
bc = (bc_left, bc_right)

u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(0.)
a = fe.dot(fe.grad(u), fe.grad(v))*fe.dx
L = f*v*fe.dx

u=fe.Function(V)
fe.solve(a == L, u, bc)

solution = fe.File("cube_solution.pvd")
solution << u


