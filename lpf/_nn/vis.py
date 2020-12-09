import numpy as np
import matplotlib.pyplot as plt


# def plot_batch_of_images(
#     images,
#     predictions,
#     y_sigma,
#     # w_sigma,
#     labels,
#     wspace=-0.6,
#     hspace=0.3,
#     figwidth=7,
#     title_pad=4,
#     fontsize=6,
#     top_margin=0.95,
# ):
#     # Inspired by
#     # https://stackoverflow.com/questions/42675864/how-to-remove-gaps-between-images-in-matplotlib

#     # images = torch.randn(5, 1, 32, 64)
#     # labels = torch.randn(5)
#     # predictions = torch.randn(5)

#     def previous_perfect_square(n):
#         return np.floor(np.sqrt(n)) ** 2

#     batch_size = len(images)
#     num_images = int(previous_perfect_square(batch_size))
#     grid_shape = int(num_images ** 0.5)

#     images = images[:num_images]
#     labels = labels[:num_images]
#     predictions = predictions[:num_images]

#     height = grid_shape * images.shape[-2]
#     width = grid_shape * images.shape[-1]

#     figwidth = figwidth * grid_shape
#     figheight = figwidth * height / width

#     figheight += hspace
#     figwidth += wspace

#     fig, axes = plt.subplots(
#         grid_shape,
#         grid_shape,
#         # gridspec_kw={'height_ratios': [32, 32], 'width_ratios': [32, 32]},
#         figsize=(figwidth, figheight),
#     )

#     def ax_imshow(ax, image):

#         one_channel = image.shape[0] == 1
#         if one_channel:
#             image = image.mean(0)
#             height, width = image.shape

#         else:
#             channels, height, width = image.shape

#         npimage = image.cpu().numpy()

#         if not one_channel:
#             npimage = np.transpose(npimage, (1, 2, 0))

#         ax.imshow(npimage)

#         return ax

#     for i, ax in enumerate(axes.flat):

#         image = images[i]
#         label = labels[i].item()
#         prediction = predictions[i].item()
#         y_unc = y_sigma[i].item()
#         w_unc = w_sigma[i].item()

#         ax_imshow(ax, image)

#         title_str = (
#             f"Target: {label:.2f}\n"
#             f"Prediction: {prediction:.2f} $\pm$ {y_unc:.2f} \n"
#             f"Weight Uncertainty: {w_unc:.2f}"
#         )

#         ax.set_title(title_str, fontsize=fontsize, pad=title_pad)
#         ax.axis("off")

#     plt.subplots_adjust(
#         wspace=wspace,
#         hspace=hspace,
#         left=0,
#         right=1,
#         bottom=1 - top_margin,
#         top=top_margin,
#     )
#     return fig


def plot_batch_of_images(
    images,
    dm_pred,
    dm_std,
    dm_t,
    fluence_pred,
    fluence_t,
    width_pred,
    width_t,
    spectral_index_pred,
    spectral_index_t,
    wspace=-0.70,
    hspace=0.4,
    figwidth=15,
    title_pad=4,
    fontsize=12,
    top_margin=0.90,
):
    # Inspired by
    # https://stackoverflow.com/questions/42675864/how-to-remove-gaps-between-images-in-matplotlib

    # images = torch.randn(5, 1, 32, 64)
    # labels = torch.randn(5)
    # predictions = torch.randn(5)

    def previous_perfect_square(n):
        return np.floor(np.sqrt(n)) ** 2

    batch_size = len(images)
    num_images = int(previous_perfect_square(batch_size))
    grid_shape = int(num_images ** 0.5)

    images = images[:num_images]
    # labels = labels[:num_images]
    # predictions = predictions[:num_images]

    height = grid_shape * images.shape[-2]
    width = grid_shape * images.shape[-1]

    figwidth = figwidth * grid_shape
    figheight = figwidth * height / width

    figheight += hspace
    figwidth += wspace

    fig, axes = plt.subplots(
        grid_shape,
        grid_shape,
        # gridspec_kw={'height_ratios': [32, 32], 'width_ratios': [32, 32]},
        figsize=(figwidth, figheight),
    )

    def ax_imshow(ax, image):

        one_channel = image.shape[0] == 1
        if one_channel:
            image = image.mean(0)
            height, width = image.shape

        else:
            channels, height, width = image.shape

        npimage = image.cpu().numpy()

        if not one_channel:
            npimage = np.transpose(npimage, (1, 2, 0))

        ax.imshow(npimage)

        return ax

    for i, ax in enumerate(axes.flat):

        ax_imshow(ax, images[i])

        title_str = (
            f"DM: {dm_t[i].item():.2f} | Prediction: {dm_pred[i].item():.2f} $\pm$ {dm_std[i].item():.2f} \n"
            f"Fluence: {fluence_t[i].item():.2f} | Prediction: {fluence_pred[i].item():.2f} \n"
            f"Width: {width_t[i].item():.2f} | Prediction: {width_pred[i].item():.2f} \n"
            f"Spectral Index: {spectral_index_t[i].item():.2f} | Prediction: {spectral_index_pred[i].item():.2f}"
        )
        ax.set_title(title_str, fontsize=fontsize, pad=title_pad)
        ax.axis("off")

    plt.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
        left=0,
        right=1,
        bottom=1 - top_margin,
        top=top_margin,
    )
    return fig