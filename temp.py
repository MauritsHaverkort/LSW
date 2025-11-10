plt.figure("coeff_arrays")
for i,(a,b,c,d) in enumerate(material.return_spatial_indices(4)):
    if np.any(np.abs(sqw_4_xyz_along_path[a,b,c,d,:])>1e-10):
        plt.plot(np.real(coeff_array_spectrum_4[a,b,c,d,:]), label=f"Re coeff 4 {a}{b}{c}{d}")
        plt.plot(np.imag(coeff_array_spectrum_4[a,b,c,d,:]), label=f"Im coeff 4 {a}{b}{c}{d}", linestyle="--")
plt.legend(ncol=10)
