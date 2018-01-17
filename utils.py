import numpy as np
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os



def delta_psi(theta1,phi1, theta2, phi2):
    sp = np.sin(theta1)*np.cos(phi1)*np.sin(theta2)*np.cos(phi2) \
         + np.sin(theta1)*np.sin(phi1)*np.sin(theta2)*np.sin(phi2) \
         +np.cos(theta1)*np.cos(theta2)
    return np.arccos(sp)


def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).
    """
    def cross_matrix(x):
        r"""Calculate cross product matrix
        A[ij] = x_i * y_j - y_i * x_j
        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(
        len(ra1) == len(dec1) == len(ra2) == len(dec2) == len(ra3) == len(dec3)
        )

    alpha = np.arccos(np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2)
                      + np.sin(dec1) * np.sin(dec2))
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1.-np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec


def setNewEdges(edges):
    newEdges = []
    for i in range(0, len(edges) - 1):
        newVal = (edges[i] + edges[i + 1]) * 1.0 / 2
        newEdges.append(newVal)
    return np.array(newEdges)

def norm_hist(h):
    h = np.array([i / np.sum(i) if np.sum(i) > 0 else i / 1. for i in h])
    return h


def create_splines(nuData, nuDataSig, zen_reco, az_reco, en_reco, spline_name):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    Hs = dict()
    
    mask_data = np.isfinite(nuData[zen_reco])
    mask = np.isfinite(nuDataSig[zen_reco])
    delta_mask = np.degrees(delta_psi(nuDataSig['trueZenith'], nuDataSig['trueAzimuth'], nuDataSig[zen_reco], nuDataSig[az_reco]))<5

    # energy ratio 2D spline
    print('Create Energy Spline..check yourself whether it is ok')
    zenith_bins=list(np.linspace(-1.,0.,20, endpoint=False)) + list(
        np.linspace(0.,1.,16))
    e_bins = np.linspace(2.0, 8, 40)

    # zenith_bins=list(np.linspace(-1.,0.,30, endpoint=False)) + list(
    #     np.linspace(0.,1.,24))
    # e_bins = np.linspace(2.0, 8, 60)

    #tot_weight = np.sum([f[flux][mask & delta_mask] for flux in ftypes], axis=0)
    #tot_weight = np.sum([nuData[flux][mask_data] for flux in ftypes, axis=0)
    tot_weight = np.sum([nuData[flux][mask_data] for flux in ['weight']], axis=0)
    
    data_x = np.cos(nuData[zen_reco][mask_data])
    data_y = nuData[en_reco][mask_data]
    sig_x = np.cos(nuDataSig[zen_reco][mask & delta_mask])
    sig_y = nuDataSig[en_reco][mask & delta_mask]

    plt.figure()
    plt.hist(data_x)
    plt.xlabel("cos(recon zenth)")
    plt.savefig("plots/cos_zen.pdf")
    plt.close()

    plt.figure()
    plt.hist(data_y)
    plt.xlabel("Recon energy")
    plt.savefig("plots/Energy.pdf")
    plt.close()

    plt.figure()
    plt.hist(sig_x)
    plt.xlabel("Cos(recon zenth)")
    plt.savefig("plots/sig_cos_zen.pdf")
    plt.close()

    plt.figure()
    plt.hist(sig_y)
    plt.xlabel("Recon energy")
    plt.savefig("plots/sig_Energy.pdf")
    plt.close()

    vals = []
    for ax in [zenith_bins, e_bins]:
        mids = []
        for i in range(1, len(ax)):
            mid = 0.5 * (ax[i] + ax[i-1])
            mids.append(mid)
        vals.append(mids)

    H_tot, xedges, yedges = np.histogram2d(data_x, data_y,
                                       #weights=tot_weight,
                                       bins=(zenith_bins, e_bins))

    H_tot_theo = np.ma.masked_array(norm_hist(H_tot))
    H_tot_theo.mask = (H_tot_theo <= 0)

    H_astro, xedges, yedges = np.histogram2d(sig_x, sig_y,
                                             weights=nuDataSig['astro'][
                                                 mask & delta_mask],
                                           bins=(zenith_bins, e_bins))

    H_astro_theo = np.ma.masked_array(norm_hist(H_astro))
    H_astro_theo.mask = (H_astro_theo <= 0)

    original_data = H_astro_theo / H_tot_theo
    original_data = np.ma.array(original_data)

    combine_mask = (H_astro_theo > 0) & (H_tot_theo > 0)
    combine_mask = np.ma.array(combine_mask)
    original_data.mask = ~combine_mask

    for i, row in enumerate(H_tot):
        new = row
        new[new == 0] = 1
        norm = np.sum(new)
        H_tot[i] = new / norm

    for i, row in enumerate(H_astro):
        new = row
        new[new == 0] = np.min(new[new > 0])
        norm = np.sum(new)
        H_astro[i] = new / norm

    plt.figure()
    ax = plt.subplot(111)
    ax.pcolormesh(vals[0], vals[1], H_tot_theo.T)
    plt.savefig("plots/energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    ax.pcolormesh(vals[0], vals[1], H_astro_theo.T)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/sig_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    data = H_tot.T
    ax.pcolormesh(vals[0], vals[1], data)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/corrected_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    ax.pcolormesh(vals[0], vals[1], H_astro.T)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/corrected_sim_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    ax.pcolormesh(vals[0], vals[1], original_data.T)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/ratio_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    log_data = np.log(original_data)
    cbar = ax.pcolormesh(vals[0], vals[1], log_data.T, vmin=-np.max(log_data),
                         vmax = np.max(log_data),
                         cmap=cm.get_cmap('seismic'))
    cbar.set_edgecolor('face')
    plt.colorbar(cbar)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/log_ratio_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    ax.pcolormesh(vals[0], vals[1], H_astro.T/H_tot.T)
    plt.savefig("plots/corrected_ratio_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    data = H_astro/H_tot
    data = np.ma.array(data)
    data.mask = data <= 0
    data = np.log(data)
    cbar = ax.pcolormesh(vals[0], vals[1], data.T, vmin=-np.max(data),
                         vmax = np.max(data),
                         cmap=cm.get_cmap('seismic'))
    cbar.set_edgecolor('face')
    plt.colorbar(cbar)
    plt.axis([zenith_bins[0], zenith_bins[-1], e_bins[0], e_bins[-1]])
    plt.savefig("plots/corrected_log_ratio_energy_vs_cos_zen.pdf")
    plt.close()

    alt_linear_spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 original_data,
                                 kx=1, ky=1, s=0)

    skylab_data = np.ma.array(original_data)

    for i, row in enumerate(original_data):
        for j, entry in enumerate(row):
            sig = isinstance(H_astro_theo[i][j], float)
            bkg = isinstance(H_tot_theo[i][j], float)

            if (sig > 0) & (bkg > 0):
                pass
            elif (sig > 0):
                skylab_data[i][j] = np.max(row)
            elif (bkg > 0):
                skylab_data[i][j] = np.min(row)
            else:
                skylab_data[i][j] = 1.
                
    # Nearest Neighbour Filling

    from scipy.interpolate import griddata

    vertical_fill = np.ones_like(original_data)

    for i, row in enumerate(original_data):
        row_mask = combine_mask[i]
        z = row
        x, y = np.meshgrid(vals[0][i], vals[1])

        points = np.vstack([x.T[0][row_mask], y.T[0][row_mask]]).T

        bins = griddata(points, z[row_mask], (x, y),
                                    method='nearest')

        for j, val in enumerate(bins):
            vertical_fill[i][j] = val

    alt_linear = np.ma.array(vertical_fill).T

    z = np.ma.array(original_data.T)

    x, y = np.meshgrid(vals[0], vals[1])

    points = zip(x[~z.mask], y[~z.mask])
    
    # Save splines

    spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 H_astro/H_tot,
                                 kx=3, ky=1, s=0)
    np.save('E_spline%s.npy'%spline_name, spline)

    alt_spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 H_astro_theo/H_tot_theo,
                                 kx=3, ky=1, s=0)



    linear_spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 H_astro/H_tot,
                                 kx=1, ky=1, s=0)

    np.save('Filled_E_spline%s.npy' % spline_name, linear_spline)

    nearest_spline = RectBivariateSpline(setNewEdges(xedges),
                                         setNewEdges(yedges),
                                         alt_linear.T,
                                         kx=1, ky=1, s=0)

    np.save('Nearest_Neighbour_E_spline%s.npy' % spline_name, nearest_spline)

    skylab_spline = RectBivariateSpline(setNewEdges(xedges),
                                 setNewEdges(yedges),
                                 skylab_data,
                                 kx=1, ky=1, s=0)

    plt.figure()
    ax = plt.subplot(111)
    cbar = ax.pcolormesh(x, y, alt_linear)
    cbar.set_edgecolor('face')
    plt.colorbar(cbar)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.savefig("plots/linear_spline_ratio_energy_vs_cos_zen.pdf")
    plt.close()

    plt.figure()
    ax = plt.subplot(111)
    data = np.log(alt_linear)
    cbar = ax.pcolormesh(x, y, data, vmin=-np.max(data), vmax=np.max(data),
                         cmap=cm.get_cmap('seismic'))
    cbar.set_edgecolor('face')
    plt.colorbar(cbar)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.savefig("plots/linear_spline_log_ratio_energy_vs_cos_zen.pdf")
    plt.close()
    
    #Plot the spline distribution

    labels = ["corrected_", "", "linear_corrected_", "original_linear_",
              "nearest_", "skylab_"]

    for i, spl in enumerate([spline, alt_spline, linear_spline,
                             alt_linear_spline, nearest_spline, skylab_spline]):

        n_dots = 100
        x = np.linspace(zenith_bins[0], zenith_bins[-1], n_dots)
        y = np.linspace(e_bins[0], e_bins[-1], n_dots)

        plt.figure()
        ax = plt.subplot(111)
        data = spl(x, y)
        data = np.ma.array(data)
        data.mask = data <= 0
        cbar = ax.pcolormesh(x, y, data.T)
        cbar.set_edgecolor('face')
        plt.colorbar(cbar)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.savefig("plots/" + labels[i] + "spline_ratio_energy_vs_cos_zen.pdf")
        plt.close()

        plt.figure()
        ax = plt.subplot(111)
        data = np.log(data)
        cbar = ax.pcolormesh(x, y, data.T, vmin=-np.max(data), vmax = np.max(data),
                             cmap=cm.get_cmap('seismic'))
        cbar.set_edgecolor('face')
        plt.colorbar(cbar)
        plt.axis([x.min(), x.max(), y.min(), y.max()])
        plt.savefig("plots/" + labels[i] + "spline_log_ratio_energy_vs_cos_zen.pdf")
        plt.close()
        
    # Attempt at KDE

    # from scipy import stats
    #
    # zenith_bins = list(np.linspace(-1., 0., 10, endpoint=False)) + list(
    #     np.linspace(0., 1., 8))
    #
    # e_bins = np.linspace(2.0, 8, 20)
    #
    # data_x = np.cos(nuData[zen_reco][mask_data])
    # data_y = nuData[en_reco][mask_data]
    #
    # sim_x = np.cos(nuDataSig[zen_reco][mask & delta_mask])
    # sim_y = nuDataSig[en_reco][mask & delta_mask]
    #
    # kde_dump = []
    #
    # for i, [x, y] in enumerate([[data_x, data_y], [sim_x, sim_y]]):
    #     xmin = np.min(x)
    #     xmax = np.max(x)
    #     ymin = np.min(y)
    #     ymax = np.max(y)
    #
    #     X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    #     positions = np.vstack([X.ravel(), Y.ravel()])
    #     values = np.vstack([x, y])
    #     kernel = stats.gaussian_kde(values)
    #     fig, ax = plt.subplots()
    #
    #     kde_vals = []
    #
    #     for zen in zenith_bins:
    #         zen_points = np.ones_like(e_bins) * zen
    #         points = np.vstack([zen_points, e_bins])
    #         values = kernel(points)
    #         norm = np.sum(values)
    #         row = values/norm
    #
    #         to_add = 10**-10
    #         norm2 = 1 + to_add * float(len(e_bins))
    #         row2 = (row + to_add)/norm2
    #         kde_vals.append(row2)
    #         # print row, norm, np.sum(row), row2, norm2, np.sum(row2)
    #
    #     kde_vals = np.ma.array(kde_vals)
    #     kde_dump.append(kde_vals)
    #
    #     del kernel
    #
    #     cbar = ax.pcolormesh(zenith_bins, e_bins, kde_vals.T)
    #     cbar.set_edgecolor('face')
    #     plt.colorbar(cbar)
    #     plt.savefig("plots/kde_" + ["", "sim_"][i] + "energy_vs_cos_zen.pdf")
    #     plt.close()
    #
    # plt.figure()
    # ax = plt.subplot(111)
    # data = kde_dump[1]/kde_dump[0]
    # data = np.ma.array(data)
    # data.mask = data <= 0
    # cbar = ax.pcolormesh(zenith_bins, e_bins, data.T, vmin = 0, vmax = 1000)
    # cbar.set_edgecolor('face')
    # plt.colorbar(cbar)
    # plt.axis([x.min(), x.max(), y.min(), y.max()])
    # plt.savefig("plots/kde_ratio_energy_vs_cos_zen.pdf")
    # plt.close()
    #
    # plt.figure()
    # ax = plt.subplot(111)
    # data = np.log(data)
    # cbar = ax.pcolormesh(zenith_bins, e_bins,  data.T, vmin=-5, vmax=5,
    #                      cmap=cm.get_cmap('seismic'))
    # cbar.set_edgecolor('face')
    # plt.colorbar(cbar)
    # plt.axis([x.min(), x.max(), y.min(), y.max()])
    # plt.savefig("plots/kde_log_ratio_energy_vs_cos_zen.pdf")
    # plt.close()
    #
    # raw_input("prompt")

    # zenith dist 1D spline
    print('Create Zenith Spline...Check if ok..')
    coszen = np.cos(nuData[zen_reco][mask_data])
    vals, edges = np.histogram(coszen,
                               #weights=tot_weight,
                               bins=30, density=True)
    print('vals')
    zen_spl = InterpolatedUnivariateSpline(setNewEdges(edges),
                                           np.log10(vals), k=1)
    print(10**zen_spl(setNewEdges(edges)))
    np.save('coszen_spl%s.npy'%spline_name, zen_spl)

    # zenith dist 1D spline
    print('Create Zenith Spline Signal with true zenith...Check if ok..')
    vals_sig, edges_sig = np.histogram(np.cos(nuDataSig['zenith'][mask & delta_mask]),
                                       weights=nuDataSig['astro'][mask & delta_mask],
                                       bins=30, density=True)
    zen_spl_sig = InterpolatedUnivariateSpline(setNewEdges(edges_sig),
                                               np.log10(vals_sig), k=1)
    print(10**zen_spl_sig(setNewEdges(edges_sig)))
    np.save('coszen_signal_spl%s.npy'%spline_name, zen_spl_sig)

    # zenith dist 1D spline
    print('Create Zenith Spline Signal with reco zenith...Check if ok..')
    vals_sig_rec, edges_sig_rec = np.histogram(np.cos(nuDataSig[zen_reco][mask & delta_mask]),
                                               weights=nuDataSig['astro'][mask & delta_mask],
                                               bins=30, density=True)
    zen_spl_sig_rec = InterpolatedUnivariateSpline(setNewEdges(edges_sig),
                                               np.log10(vals_sig), k=1)
    print(10**zen_spl_sig_rec(setNewEdges(edges_sig)))
    np.save('coszen_signal_reco_spl%s.npy'%spline_name, zen_spl_sig_rec)

    
@np.vectorize
def zen_to_dec(zen):
    return zen - 0.5*np.pi


@np.vectorize
def dec_to_zen(dec):
    return dec + 0.5*np.pi
