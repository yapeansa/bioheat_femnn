// Definindo o tamanho da malha
mesh_size = 0.04;

// Definindo pontos para o domínio externo (10x10 cm)
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {0.1, 0, 0, mesh_size};
Point(3) = {0.1, 0.1, 0, mesh_size};
Point(4) = {0, 0.1, 0, mesh_size};

// Definindo pontos para o o domínio tumoral interno (2x2 cm)
Point(5) = {0.04, 0.04, 0, mesh_size/10};
Point(6) = {0.06, 0.04, 0, mesh_size/10};
Point(7) = {0.06, 0.06, 0, mesh_size/10};
Point(8) = {0.04, 0.06, 0, mesh_size/10};

// Definindo retas para o domínio externo (fronteira)
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Definindo retas para o domínio tumoral interno
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// loop para a fronteira exterior e para a fronteira interior
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};

// Criando superfícies planas para o tecido saudável e o tecido tumoral
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};

// Definindo volumes físicos (domínios)
// Esta parte é importante para que o FEniCS identifique subdomínios
Physical Surface("healthy_tissue") = {1};
Physical Surface("tumor") = {2};

// Definindo linhas físicas (fronteiras)
// Esta parte é importante para que o FEniCS identifique condições de contorno
Physical Line("boundary_bottom") = {1};
Physical Line("boundary_right") = {2};
Physical Line("boundary_top") = {3};
Physical Line("boundary_left") = {4};