import numpy as np


# pythran export svg(complex128[:], float64[:], float64[:], complex128[:, :, :] order(C), int64[:], int64[:], float, float, float)
def svg(w_l, f_n, E_n, p_vnn, pol_v, band_n, ftol=1e-4, Etol=1e-6, eshift=0):
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
    sum2_l = np.zeros(w_l.size, dtype=np.complex128)
    sum3_l = np.zeros(w_l.size, dtype=np.complex128)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            assert nni > 0
            assert mmi > 0
            # Remove non important term using TRS
            if mmi <= nni:
                continue

            assert mmi > nni
            # Useful variables
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift

            # Comute the 2-band term
            if np.abs(Emn) > Etol and np.abs(fnm) > ftol:
                pnml = (
                    p_vnn[pol_v[0], nni, mmi]
                    * (
                        p_vnn[pol_v[1], mmi, nni]
                        * (p_vnn[pol_v[2], mmi, mmi] - p_vnn[pol_v[2], nni, nni])
                        + p_vnn[pol_v[2], mmi, nni]
                        * (p_vnn[pol_v[1], mmi, mmi] - p_vnn[pol_v[1], nni, nni])
                    )
                    / 2
                )
                sum2_l += (
                    1j
                    * fnm
                    * np.imag(pnml)
                    * (1 / (Emn ** 4 * (w_l - Emn)) - 16 / (Emn ** 4 * (2 * w_l - Emn)))
                )

            # Loop over the last band index
            for lli in band_n:
                fnl = f_n[nni] - f_n[lli]
                fml = f_n[mmi] - f_n[lli]

                # Do not do zero calculations
                if np.abs(fnl) < ftol and np.abs(fml) < ftol:
                    continue

                # Compute the susceptibility with 1/w form
                Eln = E_n[lli] - E_n[nni] + fnl * eshift
                Eml = E_n[mmi] - E_n[lli] - fml * eshift
                pnml = p_vnn[pol_v[0], nni, mmi] * (
                    p_vnn[pol_v[1], mmi, lli] * p_vnn[pol_v[2], lli, nni]
                    + p_vnn[pol_v[2], mmi, lli] * p_vnn[pol_v[1], lli, nni]
                )
                pnml = 1j * np.imag(pnml) / 2

                # Compute the divergence-free terms
                if np.abs(Emn) > Etol and np.abs(Eml) > Etol and np.abs(Eln) > Etol:
                    ftermD = (
                        (
                            16
                            / (Emn ** 3 * (2 * w_l - Emn))
                            * (fnl / (Emn - 2 * Eln) + fml / (Emn - 2 * Eml))
                        )
                        + fnl / (Eln ** 3 * (2 * Eln - Emn) * (w_l - Eln))
                        + fml / (Eml ** 3 * (2 * Eml - Emn) * (w_l - Eml))
                    )
                    sum3_l += pnml * ftermD

    return sum2_l, sum3_l
