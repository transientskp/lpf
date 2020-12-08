# type: ignore
import numpy as np
import astropy.io.fits
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def catalog_video(survey, catalog, timesteps, n_std=3):
    fps = 1


    images = [np.stack([astropy.io.fits.getdata(f).squeeze() for f in survey[t]['file']]).mean(0) for t in timesteps]
    timesteps = [catalog[t] for t in timesteps]

    dpi = matplotlib.rcParams["figure.dpi"]
    width, height = images[0].shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    t = 0
    img = plt.imshow(images[t], vmin=np.nanmean(images[t]) - n_std * np.nanstd(images[t]), vmax=np.nanmean(images[t]) + n_std * np.nanstd(images[t]))
    scat = plt.scatter(
        catalog[t]["y_peak"],
        catalog[t]["x_peak"],
        s=128,
        edgecolor="red",
        facecolor="none",
        lw=1,
    )

    def animate_func(t):
        if t % fps == 0:
            print(".", end="")
        img.set_array(images[t])
        scat.set_offsets(catalog[t][["y_peak", "x_peak"]].to_pandas())
        return [img]

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=1000 / fps,  # in ms
    )

    # anim.save(os.path.join(run_folder, "catalogue_video.mp4"))
    # HTML(anim.to_jshtml())
    # plt.close()
    return anim