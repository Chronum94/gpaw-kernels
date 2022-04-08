import numpy as np


#pythran export _sigmaloop1(complex128[:, :] order(C), float64[:], float64[:], complex128[:, :] order(C), complex128[:, :] order(C), float)
def _sigmaloop1(n_mG, deps_m, f_m, W_GG, omegat_GG, eta):
    sigma = 0.0
    dsigma = 0.0

    # init variables (is this necessary?)
    nG = n_mG.shape[1]
    deps_GG = np.empty((nG, nG))
    # sigma, dsigma = 0, 0
    sign_GG = np.empty((nG, nG), dtype=np.float64)
    x_GG = np.empty((nG, nG), dtype=np.complex128)
    dx_GG = np.empty((nG, nG), dtype=np.complex128)
    nW_G = np.empty(nG, dtype=np.complex128)
    for m in range(np.shape(n_mG)[0]):
        # sigma += 1
        # dsigma += 1
        deps_GG[:, :] = deps_m[m]
        sign_GG[:, :] = 2 * f_m[m] - 1.0

        for g1 in range(nG):
            for g2 in range(nG):
                deps_plus_omega = deps_GG[g1, g2] + omegat_GG[g1, g2]
                deps_minus_omega = deps_GG[g1, g2] - omegat_GG[g1, g2]
                
                x1_GG = 1 / (deps_plus_omega - 1.0j * eta)
                x2_GG = 1 / (deps_minus_omega + 1.0j * eta)
                x3_GG = 1 / (deps_plus_omega - 1.0j * eta * sign_GG[g1, g2])
                x4_GG = 1 / (deps_GG[g1, g2] - omegat_GG[g1, g2] - 1.0j * eta * sign_GG[g1, g2])
                
                x_GG[g1, g2] = W_GG[g1, g2] * (sign_GG[g1, g2] * (x1_GG - x2_GG) + x3_GG + x4_GG)
                dx_GG[g1, g2] = W_GG[g1, g2] * (sign_GG[g1, g2] * (x1_GG**2 - x2_GG**2) +
                                x3_GG**2 + x4_GG**2)

        nW_G[:] = np.dot(n_mG[m], x_GG)
        sigma += np.dot(n_mG[m].conj(), nW_G).real
        nW_G[:] = np.dot(n_mG[m], dx_GG)
        dsigma -= np.dot(n_mG[m].conj(), nW_G).real
    return sigma, dsigma


# #pythran export _sigmaloop2(complex128[:, :] order(C), float64[:], float64[:], complex128[:, :] order(C), complex128[:, :] order(C), float)
# def _sigmaloop2(n_mG, deps_m, f_m, W_GG, omegat_GG, eta):
#     sigma = 0.0
#     dsigma = 0.0

#     # init variables (is this necessary?)
#     nG = n_mG.shape[1]
#     # deps_GG = np.empty((nG, nG))
#     sigma, dsigma = 0, 0
#     # sign_GG = np.empty((nG, nG), dtype=np.float64)
#     x_GG = np.empty((nG, nG), dtype=np.complex128)
#     dx_GG = np.empty((nG, nG), dtype=np.complex128)
#     nW_G = np.empty(nG)
#     for m in range(np.shape(n_mG)[0]):
#         # sigma += 1
#         # dsigma += 1
#         deps_GG = deps_m[m]
#         sign_GG = 2 * f_m[m] - 1

#         for g1 in range(nG):
#             for g2 in range(nG):
#                 deps_plus_omega = deps_GG + omegat_GG[g1, g2]
#                 deps_minus_omega = deps_GG - omegat_GG[g1, g2]
                
#                 x1_GG = 1 / (deps_plus_omega - 1j * eta)
#                 x2_GG = 1 / (deps_minus_omega + 1j * eta)
#                 x3_GG = 1 / (deps_plus_omega - 1j * eta * sign_GG)
#                 x4_GG = 1 / (deps_minus_omega - 1j * eta * sign_GG)
                
#                 x_GG[g1, g2] = W_GG[g1, g2] * (sign_GG * (x1_GG - x2_GG) + x3_GG + x4_GG)
#                 dx_GG[g1, g2] = W_GG[g1, g2] * (sign_GG * (x1_GG**2 - x2_GG**2) +
#                                 x3_GG**2 + x4_GG**2)

#         nW_G[:] = np.dot(n_mG[m], x_GG)
#         sigma += np.dot(n_mG[m].conj(), nW_G).real
#         nW_G[:] = np.dot(n_mG[m], dx_GG)
#         dsigma -= np.dot(n_mG[m].conj(), nW_G).real
#     return sigma, dsigma
