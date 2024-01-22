Credit to Iman Datta

The CSV files in this directory give spline fits for various 1D parameters used in cases A0-A4 as described in:
Vogman et. al., "Two-fluid and kinetic transport physics of Kelvin–Helmholtz instabilities in nonuniform low-beta plasmas", Phys.Plasmas 27, 102109 (2020); https://doi.org/10.1063/5.0014489.

These are 1D functions phi(x), ni_aux(x), Ex(x), Bz(x).  phi(x) and ni_aux(x) are needed to construct ion and electron distribution functions for equalibrium initial conditions in these cases.  Ex(x) and Bz(x) define the corresponding Maxwell field equilibrium initial conditions.

These fits are of the form of a spline S(x) = sum_j=1^K a_j * psi(|x-x_j|) + b_1 + b_2 * x, where psi(r)=r^3, as described in Appendix B of:
Vogman et. al., "Customizable two-species kinetic equilibria for nonuniform low-beta plasmas", Phys. Plasmas 26, 042119 (2019); https://doi.org/10.1063/1.5089465

So each CSV file is for a different function (phi, ni_aux, Ex, Bz) for a particular case (A0-A4).
Column 1 is the vector of x_j.
Column 2 is the vector of a_j.
Column 3 is the vector of b_j.  Note, only the first 2 values constitute b_j.  The rest are padding zeros to conform with CSV needing columns to be equivalent lengths.

For phi, Ex, and Bz, the size of x_j and a_j = 256.  For ni_aux, ghost nodes were used to handle the boundary conditions (10 on each side), so their size should be 276.
