import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.io import fits
import casacore.tables as tables
from viscube import sigma_by_baseline_scan_time_diff
from viscube.grid_cube import grid_cube_all_stats_wbinned
from typing import Tuple

def to_npol_nchan_nrow(arr, npol_expected, nchan_expected, nrow_expected=None, name="array"):
    """
    Convert casacore getcol output to (npol, nchan, nrow).

    Common shapes seen in the wild:
      - (nchan, npol, nrow)
      - (npol, nchan, nrow)
      - (nrow, nchan, npol)   <-- your case
      - (nrow, npol, nchan)
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} expected 3D, got {arr.shape}")

    s0, s1, s2 = arr.shape

    # helper to sanity check optional nrow
    def _check_nrow(nrow_axis_len):
        return (nrow_expected is None) or (nrow_axis_len == nrow_expected)

    # (nchan, npol, nrow) -> (npol, nchan, nrow)
    if (s0 == nchan_expected) and (s1 == npol_expected) and _check_nrow(s2):
        return np.transpose(arr, (1, 0, 2))

    # already (npol, nchan, nrow)
    if (s0 == npol_expected) and (s1 == nchan_expected) and _check_nrow(s2):
        return arr

    # (nrow, nchan, npol) -> (npol, nchan, nrow)
    if _check_nrow(s0) and (s1 == nchan_expected) and (s2 == npol_expected):
        return np.transpose(arr, (2, 1, 0))

    # (nrow, npol, nchan) -> (npol, nchan, nrow)
    if _check_nrow(s0) and (s1 == npol_expected) and (s2 == nchan_expected):
        return np.transpose(arr, (1, 2, 0))

    raise ValueError(
        f"{name} has unexpected shape {arr.shape}; expected a permutation of "
        f"(npol, nchan, nrow)=({npol_expected}, {nchan_expected}, nrow)"
        + (f" with nrow={nrow_expected}" if nrow_expected is not None else "")
    )

def hermitian_augment_w(
    u0: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    vis0: np.ndarray,
    weights0: np.ndarray,
    invvar_re0: np.ndarray,
    invvar_im0: np.ndarray,
):
    """
    Hermitian augment:
      (u, v, w, Re, Im, wgt, invvar_re, invvar_im)
      -> concat with
      (-u, -v, -w, +Re, -Im, wgt, invvar_re, invvar_im)

    Returns
    -------
    uu, vv, ww, vis_re, vis_imag, wgt, invvar_re_aug, invvar_im_aug
    """
    # Hermitian augment in u,v,w + vis (IMPORTANT: w must flip sign too)
    uu = np.concatenate([u0, -u0], axis=0)
    vv = np.concatenate([v0, -v0], axis=0)
    ww = np.concatenate([w0, -w0], axis=0)
    vis_re = np.concatenate([vis0.real,  vis0.real], axis=0)
    vis_im = np.concatenate([vis0.imag, -vis0.imag], axis=0)
    wgt = np.concatenate([weights0, weights0], axis=0)

    # Variance does not change under sign flip/conjugation
    invvar_re_aug = np.concatenate([invvar_re0, invvar_re0], axis=0)
    invvar_im_aug = np.concatenate([invvar_im0, invvar_im0], axis=0)

    return uu, vv, ww, vis_re, vis_im, wgt, invvar_re_aug, invvar_im_aug

def gridder_AIO(jobname, msfolder, outputfolder):

    ms_path = "./"+msfolder+"/Lsim_"+jobname+".ms"  #path to your measurement set

    assert os.path.exists(ms_path), f"MS not found: {ms_path}"
    dd_tab  = tables.table(os.path.join(ms_path, "DATA_DESCRIPTION"), readonly=True)
    spw_tab = tables.table(os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=True)
    pol_tab = tables.table(os.path.join(ms_path, "POLARIZATION"), readonly=True)

    dd_spw_ids = dd_tab.getcol("SPECTRAL_WINDOW_ID")   # (n_ddid,)
    dd_pol_ids = dd_tab.getcol("POLARIZATION_ID")      # (n_ddid,)

    ddids = np.arange(dd_spw_ids.size)

    for ddid in ddids:
        spw_id = int(dd_spw_ids[ddid])
        chan_freq = spw_tab.getcell("CHAN_FREQ", spw_id)  # (nchan,)
        nchan = len(chan_freq)

    dd_tab.close()
    spw_tab.close()
    pol_tab.close()

    datadescid = 0  # <-- set this to the ddid you want (single SPW setup)
    data_column = "DATA"  # or "CORRECTED_DATA", this part depends on the preprocessing pipeline you used

    t = tables.table(ms_path, readonly=True)

    t_sel = t.query(f"DATA_DESC_ID=={int(datadescid)}")

    nrow = t_sel.nrows()

    ant1 = t_sel.getcol("ANTENNA1", 0, nrow)         # (nrow,)
    ant2 = t_sel.getcol("ANTENNA2", 0, nrow)         # (nrow,)
    uvw_m = t_sel.getcol("UVW", 0, nrow)             # often (3, nrow) in meters
    flag_row = t_sel.getcol("FLAG_ROW", 0, nrow)     # (nrow,)

    time_s = t_sel.getcol("TIME", 0, nrow)           # (nrow,)
    scan   = t_sel.getcol("SCAN_NUMBER", 0, nrow)    # (nrow,)

    flag = t_sel.getcol("FLAG", 0, nrow)

    weight = t_sel.getcol("WEIGHT", 0, nrow)

    data = t_sel.getcol(data_column, 0, nrow)

    t_sel.close()
    t.close()

    dd_tab  = tables.table(os.path.join(ms_path, "DATA_DESCRIPTION"), readonly=True)
    spw_tab = tables.table(os.path.join(ms_path, "SPECTRAL_WINDOW"), readonly=True)
    pol_tab = tables.table(os.path.join(ms_path, "POLARIZATION"), readonly=True)

    spw_id = int(dd_tab.getcol("SPECTRAL_WINDOW_ID")[datadescid])
    pol_id = int(dd_tab.getcol("POLARIZATION_ID")[datadescid])

    chan_freq_hz = np.array(spw_tab.getcell("CHAN_FREQ", spw_id), dtype=float).flatten()  # (nchan,)
    corr_type = np.array(pol_tab.getcell("CORR_TYPE", pol_id)).flatten()
    npol = corr_type.size
    nchan = chan_freq_hz.size

    dd_tab.close()
    spw_tab.close()
    pol_tab.close()

    data = to_npol_nchan_nrow(data, npol, nchan, nrow_expected=nrow, name=data_column)
    flag = to_npol_nchan_nrow(flag, npol, nchan, nrow_expected=nrow, name="FLAG")

    uvw_m = np.asarray(uvw_m)
    if uvw_m.shape == (nrow, 3):
        uvw_m = uvw_m.T
    assert uvw_m.shape == (3, nrow), f"Unexpected UVW shape: {uvw_m.shape}"

    weight = np.asarray(weight)

    if weight.ndim == 1:
        assert weight.shape[0] == nrow
        weight = np.tile(weight[None, :], (npol, 1))

    elif weight.ndim == 2:
        if weight.shape == (npol, nrow):
            pass
        elif weight.shape == (nrow, npol):
            weight = weight.T
        elif weight.shape[0] == 1 and weight.shape[1] == nrow and npol > 1:
            weight = np.tile(weight, (npol, 1))
        else:
            raise ValueError(f"Unexpected WEIGHT shape: {weight.shape}")
    else:
        raise ValueError(f"Unexpected WEIGHT ndim: {weight.ndim}, shape: {weight.shape}")

    flag = np.logical_or(flag, flag_row[None, None, :])

    xc = np.where(ant1 != ant2)[0]

    data = data[:, :, xc]
    flag = flag[:, :, xc]
    uvw_m = uvw_m[:, xc]
    weight = weight[:, xc]

    time_s = time_s[xc]
    scan   = scan[xc]
    ant1   = ant1[xc]
    ant2   = ant2[xc]

    w_b = weight[:, None, :]                  # (npol, 1, nvis)
    wsum = np.sum(w_b, axis=0)                # (nchan, nvis)
    wsum_safe = np.where(wsum > 0, wsum, 1.0)

    data = np.sum(data * w_b, axis=0) / wsum_safe   # (nchan, nvis)
    flag = np.any(flag, axis=0)                     # (nchan, nvis)
    weight_row = np.sum(weight, axis=0)             # (nvis,)

    time_row = time_s
    scan_row = scan
    ant1_row = ant1
    ant2_row = ant2

    mask = ~flag  # (nchan, nvis)

    u_m, v_m, w_m = uvw_m
    nu = chan_freq_hz[:, None]  # (nchan, 1)

    u_lam = (u_m[None, :] * nu / c.value)  # (nchan, nvis)
    v_lam = (v_m[None, :] * nu / c.value)  # (nchan, nvis)
    w_lam = (w_m[None, :] * nu / c.value)  # (nchan, nvis)

    is_increasing = np.all(np.diff(chan_freq_hz) > 0)

    if not is_increasing:
        chan_freq_work_hz = chan_freq_hz[::-1].copy()
        data_work = data[::-1].copy()
        mask_work = mask[::-1].copy()
        u_work = u_lam[::-1].copy()
        v_work = v_lam[::-1].copy()
        w_work = w_lam[::-1].copy()
    else:
        chan_freq_work_hz = chan_freq_hz.copy()
        data_work = data.copy()
        mask_work = mask.copy()
        u_work = u_lam.copy()
        v_work = v_lam.copy()
        w_work = w_lam.copy()

    n_avg = 1  # fixed per your preference

    nchan = data_work.shape[0]
    n_new = (nchan // n_avg) * n_avg

    data_tr = data_work[:n_new]
    u_tr = u_work[:n_new]
    v_tr = v_work[:n_new]
    w_tr = w_work[:n_new]
    mask_tr = mask_work[:n_new]
    freq_tr = chan_freq_work_hz[:n_new]

    data_avg = data_tr.reshape(n_new // n_avg, n_avg, -1).mean(axis=1)
    u_avg = u_tr.reshape(n_new // n_avg, n_avg, -1).mean(axis=1)
    v_avg = v_tr.reshape(n_new // n_avg, n_avg, -1).mean(axis=1)
    w_avg = w_tr.reshape(n_new // n_avg, n_avg, -1).mean(axis=1)

    mask_avg = mask_tr.astype(float).reshape(n_new // n_avg, n_avg, -1).mean(axis=1) > 0.5

    chan_freq_avg_hz = freq_tr.reshape(n_new // n_avg, n_avg).mean(axis=1)

    sigma_re_avg, sigma_im_avg = sigma_by_baseline_scan_time_diff(
        data_avg, mask_avg,
        time_row=time_row,
        scan_row=scan_row,
        ant1_row=ant1_row,
        ant2_row=ant2_row,
        min_pairs=8,
    )

    invvar_group_re_avg = np.where(
        mask_avg,
        1.0 / np.maximum(sigma_re_avg, 1e-12)**2,
        0.0
    )
    invvar_group_im_avg = np.where(
        mask_avg,
        1.0 / np.maximum(sigma_im_avg, 1e-12)**2,
        0.0
    )

    valid = mask_avg  # (nchan_avg, nvis)

    u_cont = u_avg[valid]
    v_cont = v_avg[valid]
    w_cont = w_avg[valid]
    vis_cont = data_avg[valid]

    weight_chan = np.tile(weight_row[None, :], (data_avg.shape[0], 1)).astype(float)
    wt_cont = weight_chan[valid]

    sigma_re_cont = sigma_re_avg[valid]
    sigma_im_cont = sigma_im_avg[valid]
    invvar_group_re_cont = invvar_group_re_avg[valid]
    invvar_group_im_cont = invvar_group_im_avg[valid]

    u0 = u_cont
    v0 = v_cont
    w0 = w_cont
    vis0 = vis_cont
    weight0 = wt_cont
    frequencies = chan_freq_avg_hz

    sigma_re0 = sigma_re_cont
    sigma_im0 = sigma_im_cont
    invvar_re0 = invvar_group_re_cont
    invvar_im0 = invvar_group_im_cont

    cell_size =  1.5 # arcsec
    npix = 5200
    pad_uv = 0.0
    FOV_arcsec = cell_size * npix # arcsec

    uu, vv, ww, vis_re, vis_im, wt, invvar_re_aug, invvar_im_aug = hermitian_augment_w(
        u0, v0, w0, vis0, weight0, invvar_re0, invvar_im0
    )

    UU = uu[None, :]
    VV = vv[None, :]
    WW = ww[None, :]
    RE = vis_re[None, :]
    IM = vis_im[None, :]
    WG = wt[None, :]
    # SR = sigma_re_aug[None, :]
    # SI = sigma_im_aug[None, :]
    IR = invvar_re_aug[None, :]
    II = invvar_im_aug[None, :]

    w_bins = 12         # number of w bins
    w_abs = False       # if True bins |w| (often increases per-bin counts)

    window_name = "kaiser_bessel"
    window_kwargs = {"m": 6}

    std_min_effective = 10

    mean_re_w, mean_im_w, std_re_w, std_im_w, counts_w, u_edges, v_edges, w_edges = grid_cube_all_stats_wbinned(
        frequencies=np.array([0.0]),  # unused
        uu=UU,
        vv=VV,
        ww=WW,
        vis_re=RE,
        vis_imag=IM,
        weight=WG,
        invvar_group_re=IR,
        invvar_group_im=II,
        npix=npix,
        pad_uv=pad_uv,
        fov_arcsec = FOV_arcsec,
        w_bins=w_bins,
        w_abs=w_abs,
        window_name=window_name,
        window_kwargs=window_kwargs,
        p_metric=1,
        std_min_effective=std_min_effective
    )

    uvw_grid_npz = f"./"+outputfolder+"/uvw_gridded_"+jobname+".npz"
    np.savez_compressed(
        uvw_grid_npz,
        mean_re=mean_re_w,
        mean_im=mean_im_w,
        std_re=std_re_w,
        std_im=std_im_w,
        counts=counts_w,
        u_edges=u_edges,
        v_edges=v_edges,
        w_edges=w_edges,
    )

    counts_sum = np.sum(counts_w[0], axis=0)  # (Nu, Nv)
    counts_sum = np.where(counts_sum > 0, counts_sum, 1.0)

    uv_re_collapse = np.sum(mean_re_w[0] * counts_w[0], axis=0) / counts_sum
    uv_im_collapse = np.sum(mean_im_w[0] * counts_w[0], axis=0) / counts_sum

    combined_vis_collapse = uv_re_collapse + 1j * uv_im_collapse

    fov_arcseconds = FOV_arcsec
    arcseconds_per_pixel = fov_arcseconds / npix
    x_arcsec = (np.arange(npix) - (npix // 2)) * arcseconds_per_pixel
    y_arcsec = (np.arange(npix) - (npix // 2)) * arcseconds_per_pixel

    dirty_image_collapse = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(combined_vis_collapse)))
    dirty_image_collapse = np.abs(dirty_image_collapse)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        dirty_image_collapse,
        extent=[x_arcsec[0], x_arcsec[-1], y_arcsec[0], y_arcsec[-1]],
        origin="lower",
        cmap="inferno", vmax = 0.003
    )
    plt.colorbar(label="Intensity")
    plt.title("Dirty Image from UVW-gridding (collapsed over w-bins)")
    plt.xlabel("Arcseconds")
    plt.ylabel("Arcseconds")
    plt.tight_layout()
    plt.savefig("./"+outputfolder+"/dirty_image_"+jobname+".jpg", dpi=300)
    plt.close()
    
    fits.writeto("./"+outputfolder+"/dirty_image_"+jobname+".fits", dirty_image_collapse.astype(np.float32), overwrite=True)

if __name__ == "__main__":
    jobname = sys.argv[1]
    
    msfolder = "jvlaatm"
    outputfolder = "jvlaagrid"
    gridder_AIO(jobname, msfolder, outputfolder)
    
    msfolder = "jvlabtm"
    outputfolder = "jvlabgrid"
    gridder_AIO(jobname, msfolder, outputfolder)
    
    msfolder = "lofartm"
    outputfolder = "lofargrid"
    gridder_AIO(jobname, msfolder, outputfolder)
