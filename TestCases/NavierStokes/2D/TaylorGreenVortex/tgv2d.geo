SetFactory("Built-in");

pi = Pi;
nx = 32;
ny = 32;

// Corner points
Point(1) = {-pi, -pi, 0, 1.0};
Point(2) = { pi, -pi, 0, 1.0};
Point(3) = { pi,  pi, 0, 1.0};
Point(4) = {-pi,  pi, 0, 1.0};

// Boundary curves
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Structured mesh: 32 cells => 33 points on each side
Transfinite Curve {1, 3} = nx + 1;
Transfinite Curve {2, 4} = ny + 1;
Transfinite Surface {1};

// Recombine triangles into quads
Recombine Surface {1};

// Periodicity
// right boundary = left boundary translated by (+2*pi, 0, 0)
Periodic Curve {2} = {4} Translate {2*pi, 0, 0};

// top boundary = bottom boundary translated by (0, +2*pi, 0)
Periodic Curve {3} = {1} Translate {0, 2*pi, 0};

// Optional physical groups
Physical Surface("domain", 1) = {1};
Physical Curve("xminus", 2) = {4};
Physical Curve("xplus", 3)  = {2};
Physical Curve("yminus", 4) = {1};
Physical Curve("yplus", 5)  = {3};
