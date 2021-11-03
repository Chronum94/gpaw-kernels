# These are two Python (Pythran) kernels for the calculation of second-harmonic generation in GPAW.

## How this works:

1. Copy the two files, slg.py and svg.py into the gpaw/nlopt/ directory.
2. Compile both with `pythran <filename> -Ofast`. The Ofast is necessary here for >10-20x speedups, since it gives freedom to the compiler to change the reduction order in `sum2_l` and `sum3_l`.
4. Add `from .svg import svg` and `from .slg import slg` near the top of the shg.py file in the gpaw/nlopt/ directory.
5. Add `USE_PYTHRAN_KERNEL = True` near the top of the file.
6. Change the `shg_velocity_gauge`  and `shg_length_gauge` functions to:

```python

def shg_velocity_gauge(
        w_l, f_n, E_n, p_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing in velocity gauge

    Input:
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        p_vnn           Momentum matrix elements
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        eshift          Bandgap correction
    Output:
        sum2_l, sum3_l  Output 2 and 3 bands terms
    """

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    # print(f_n)

    if USE_PYTHRAN_KERNEL:
        band_n = np.array(band_n)

        return svg(w_l, f_n.astype(float), E_n.astype(float), p_vnn, np.array(pol_v),
                   band_n, ftol, Etol, eshift)
```

```python
def shg_length_gauge(
        w_l, f_n, E_n, r_vnn, rd_vvnn, D_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing in length gauge

    Input:
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        r_vnn           Momentum matrix elements
        rd_vvnn         Generalized derivative of position
        D_vnn           Velocity difference
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        eshift          Bandgap correction
    Output:
        sum2_l, sum3_l  Output 2 and 3 bands terms
    """

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))

    if USE_PYTHRAN_KERNEL:
        band_n = np.array(band_n)

        return slg(w_l, f_n.astype(float), E_n.astype(float), r_vnn, rd_vvnn, D_vnn, np.array(pol_v), 
                   band_n, ftol, Etol, eshift)
```

6. Test run the kernel and compare results with the baseline Python implementation.
7. Typical speedups are of the order of 20x


Things that can almost certainly be done better:

[ ] - Blocked summation of the terms? I don't know if Pythran does this by default.
[ ] - Changing of order loop indices?
