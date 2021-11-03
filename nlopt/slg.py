import numpy as np


# pythran export slg(complex128[:], float64[:], float64[:], complex128[:, :, :] order(C), complex128[:, :, :, :] order(C), complex128[:, :, :] order(C), int64[:], int64[:], float, float, float)
def slg(
    w_l,
    f_n,
    E_n,
    r_vnn,
    rd_vvnn,
    D_vnn,
    pol_v,
    band_n=None,
    ftol=1e-4,
    Etol=1e-6,
    eshift=0,
):
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
    sum2_l = np.zeros(w_l.size, dtype=np.complex128)
    sum3_l = np.zeros(w_l.size, dtype=np.complex128)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Remove the non important term using TRS
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift

            # Two band part
            if np.abs(fnm) > ftol:
                tmp = (
                    2
                    * np.imag(
                        r_vnn[pol_v[0], nni, mmi]
                        * (
                            rd_vvnn[pol_v[1], pol_v[2], mmi, nni]
                            + rd_vvnn[pol_v[2], pol_v[1], mmi, nni]
                        )
                    )
                    / (Emn * (2 * w_l - Emn))
                )
                tmp += np.imag(
                    r_vnn[pol_v[1], mmi, nni] *
                    rd_vvnn[pol_v[2], pol_v[0], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni] *
                    rd_vvnn[pol_v[1], pol_v[0], nni, mmi]
                ) / (Emn * (w_l - Emn))
                tmp += (
                    np.imag(
                        r_vnn[pol_v[0], nni, mmi]
                        * (
                            r_vnn[pol_v[1], mmi, nni] *
                            D_vnn[pol_v[2], mmi, nni]
                            + r_vnn[pol_v[2], mmi, nni] *
                            D_vnn[pol_v[1], mmi, nni]
                        )
                    )
                    * (1 / (w_l - Emn) - 4 / (2 * w_l - Emn))
                    / Emn ** 2
                )
                tmp -= np.imag(
                    r_vnn[pol_v[1], mmi, nni] *
                    rd_vvnn[pol_v[0], pol_v[2], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni] *
                    rd_vvnn[pol_v[0], pol_v[1], nni, mmi]
                ) / (2 * Emn * (w_l - Emn))
                sum2_l += 1j * fnm * tmp / 2  # 1j imag

            # Three band term
            for lli in band_n:
                fnl = f_n[nni] - f_n[lli]
                fml = f_n[mmi] - f_n[lli]
                Eml = E_n[mmi] - E_n[lli] - fml * eshift
                Eln = E_n[lli] - E_n[nni] + fnl * eshift
                # Do not do zero calculations
                if np.abs(fnm) < ftol and np.abs(fnl) < ftol and np.abs(fml) < ftol:
                    continue
                if np.abs(Eln - Eml) < Etol:
                    continue

                rnml = np.real(
                    r_vnn[pol_v[0], nni, mmi]
                    * (
                        r_vnn[pol_v[1], mmi, lli] * r_vnn[pol_v[2], lli, nni]
                        + r_vnn[pol_v[2], mmi, lli] * r_vnn[pol_v[1], lli, nni]
                    )
                ) / (2 * (Eln - Eml))
                if np.abs(fnm) > ftol:
                    sum3_l += 2 * fnm / (2 * w_l - Emn) * rnml
                if np.abs(fnl) > ftol:
                    sum3_l += -fnl / (w_l - Eln) * rnml
                if np.abs(fml) > ftol:
                    sum3_l += fml / (w_l - Eml) * rnml

    return sum2_l, sum3_l
