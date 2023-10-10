from fenics import *

# Create mesh
mesh = UnitSquareMesh.create(64, 64, CellType.Type.quadrilateral)

# Next, we need to define the function space. ::

# Build function space with Lagrange multiplier
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, P1 * R)


# Define variational problem
(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = (inner(grad(u), grad(v)) + c*v + u*d)*dx
L = f*v*dx + g*v*ds

# 
# To compute the solution we use the bilinear form, the linear forms,
# and the boundary condition, but we also need to create a
# :py:class:`Function <dolfin.functions.function.Function>` to store the
# solution(s).  The (full) solution will be stored in ``w``, which we
# initialize using the
# :py:class:`FunctionSpace<dolfin.functions.functionspace.FunctionSpace>`
# ``W``.  The actual computation is performed by calling
# :py:func:`solve<dolfin.fem.solving.solve>`.  The separate components
# ``u`` and ``c`` of the solution can be extracted by calling the split
# function.  Finally, we output the solution to a ``VTK`` file to examine the result. ::

# Compute solution
w = Function(W)
solve(a == L, w)
(u, c) = w.split()

# Save solution in VTK format
file = File("neumann_poisson.pvd")
file << u
