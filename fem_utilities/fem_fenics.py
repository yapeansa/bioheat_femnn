import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import XDMFFile
from mpi4py import MPI
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import pyvista as pv

class BioHeatSimulation:
    def __init__(self, mesh_path="mesh/malha.xdmf"):
        # 1. Leitura da Malha e Domínio
        with XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
            self.domain = xdmf.read_mesh(name="Grid")
            self.subdomains = xdmf.read_meshtags(self.domain, name="Grid")

        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        
        # 2. Parâmetros Físicos
        self.k_saudavel, self.k_tumoral = 0.5, 0.55
        self.wb_saudavel, self.wb_tumoral = 0.00051, 0.00125
        self.Qm_saudavel, self.Qm_tumoral = 420.0, 4200.0
        self.rhob, self.cb, self.Ta = 1000.0, 4200.0, 37.0
        self.r0 = 3.1e-3

        # 3. Configuração de Propriedades por Subdomínio
        self._setup_properties()
        
        # 4. Condições de Contorno
        self._setup_boundary_conditions()

    def _setup_properties(self):
        V_dg = fem.functionspace(self.domain, ("DG", 0))
        self.k = fem.Function(V_dg)
        self.wb = fem.Function(V_dg)
        self.Qm = fem.Function(V_dg)

        cells_1 = self.subdomains.find(1)
        cells_2 = self.subdomains.find(2)

        self.k.x.array[cells_1] = self.k_saudavel
        self.k.x.array[cells_2] = self.k_tumoral
        self.wb.x.array[cells_1] = self.wb_saudavel
        self.wb.x.array[cells_2] = self.wb_tumoral
        self.Qm.x.array[cells_1] = self.Qm_saudavel
        self.Qm.x.array[cells_2] = self.Qm_tumoral

    def _setup_boundary_conditions(self):
        def left_boundary(x):
            return np.isclose(x[0], 0)

        boundary_facets = mesh.locate_entities_boundary(
            self.domain, self.domain.topology.dim - 1, left_boundary
        )
        dofs = fem.locate_dofs_topological(
            self.V, self.domain.topology.dim - 1, boundary_facets
        )
        self.bc = fem.dirichletbc(PETSc.ScalarType(self.Ta), dofs, self.V)

    def laser_source_eval(self, x, pontos, A_val):
        total_qr = np.zeros(x.shape[1])
        for x0, y0 in pontos:
            r2 = (x[0] - x0)**2 + (x[1] - y0)**2
            total_qr += A_val * np.exp(-r2 / self.r0**2)
        return total_qr

    def extract_nodes(self):
        return self.V.tabulate_dof_coordinates()[:, :2]

    def extract_system_matrices(self, pontos_laser, A_val):
        T = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        Qr = fem.Function(self.V)
        Qr.interpolate(lambda x: self.laser_source_eval(x, pontos_laser, A_val))
        
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.subdomains)

        a_ufl = (self.k * ufl.dot(ufl.grad(T), ufl.grad(v)) * dx + 
                 self.rhob * self.cb * self.wb * T * v * dx)
        L_ufl = (self.rhob * self.cb * self.wb * self.Ta + self.Qm + Qr) * v * dx

        a = fem.form(a_ufl)
        L = fem.form(L_ufl)

        # Montagem Matriz A
        A = assemble_matrix(a, bcs=[self.bc])
        A.assemble()
        
        # Montagem Vetor b
        b = assemble_vector(L)
        b.assemble()
        apply_lifting(b, [a], bcs=[[self.bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [self.bc])
        b.assemble()
        
        # Conversão CSR
        ai, aj, av = A.getValuesCSR()
        A_csr = csr_matrix((av, aj, ai), shape=A.getSize())
        return A_csr, b.array.copy()

    def run_simulation(self, pontos_laser, A_val, filename=None):
        T = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        Qr = fem.Function(self.V)
        Qr.interpolate(lambda x: self.laser_source_eval(x, pontos_laser, A_val))
        
        dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.subdomains)

        a = (self.k * ufl.dot(ufl.grad(T), ufl.grad(v)) * dx + 
             self.rhob * self.cb * self.wb * T * v * dx)
        L = (self.rhob * self.cb * self.wb * self.Ta + self.Qm + Qr) * v * dx

        problem = LinearProblem(a, L, bcs=[self.bc], 
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                petsc_options_prefix="solver_")
        T_sol = problem.solve()

        print(f"\nSimulação Concluída")
        print(f"Temperatura Máxima: {np.max(T_sol.x.array):.2f} °C")

        if filename:
            with io.XDMFFile(self.domain.comm, filename, "w") as file:
                file.write_mesh(self.domain)
                file.write_function(T_sol)
        
        return T_sol

    def plot_solution(self, T_input):
        if isinstance(T_input, np.ndarray):
            T_sol = fem.Function(self.V)
            T_sol.x.array[:] = T_input.flatten()
        else:
            T_sol = T_input
        
        topology, cell_types, x = plot.vtk_mesh(self.domain)
        grid = pv.UnstructuredGrid(topology, cell_types, x)
        grid.point_data["Temperatura (°C)"] = T_sol.x.array
        
        plotter = pv.Plotter()
        sargs = dict(title="Temperatura (°C)\n", vertical=True, position_x=0.85, 
                     position_y=0.15, height=0.7, width=0.04, title_font_size=19)
        
        plotter.add_mesh(grid, show_edges=False, cmap="viridis", scalar_bar_args=sargs)
        plotter.view_xy()
        plotter.show()
