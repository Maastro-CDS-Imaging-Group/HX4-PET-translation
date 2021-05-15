import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


class NdimageVisualizer():
    def __init__(self):
        # Everything in sitk (W,H,D) format
        self.spacing = (1.0, 1.0, 3.0)

        self.suv_window = {'level':1, 'width':3}
        self.hu_window = {'level':0, 'width':300}

        self.cmap_dict = {'PET': 'inferno', 'CT': 'gray', 'labelmap': 'gray', 'normalized': 'gray'}
        self.dpi = 80

    def set_spacing(self, spacing):
        self.spacing = spacing

    def set_suv_window(self, window):
        self.suv_window = window

    def set_hu_window(self, window):
        self.hu_window = window

    def _custom_imshow(self, ax, image, title, image_type):
        # Apply window
        if image_type == 'labelmap' or image_type == 'normalized':
            ax.imshow(image, cmap=self.cmap_dict[image_type])

        else:
            if image_type == 'PET':
                window = self.suv_window
            elif image_type == 'CT':
                window = self.hu_window
            win_min = window['level'] - window['width'] // 2
            win_max = window['level'] + window['width'] // 2
            ax.imshow(image, cmap=self.cmap_dict[image_type], vmin=win_min, vmax=win_max)

        ax.set_title(title)
        ax.axis('off')


    def multi_image_strips(self, image_np_list, image_types, idx_range, view='axial', subtitles=[], title=""):
        array_size = image_np_list[0].shape
        phy_size = [int(array_size[i]*self.spacing[i]) for i in range(3)]

        n_images = len(image_np_list)
        figsize = (n_images*450)/self.dpi, ((idx_range[1]-idx_range[0])*450)/self.dpi
        fig, axs = plt.subplots(1, n_images, figsize=figsize)

        if len(subtitles) != n_images: subtitles = image_types

        if view == 'axial':
            for i, image_np in enumerate(image_np_list):
                slice_list = []
                for j, s in enumerate(range(*idx_range)):
                    axial_slice = image_np[:, :, s].T
                    slice_list.append(axial_slice)
                strip = np.concatenate(slice_list, axis=0)
                self._custom_imshow(axs[i], strip, title=subtitles[i], image_type=image_types[i])

        if view == 'coronal':
            for i, image_np in enumerate(image_np_list):
                slice_list = []
                for j, s in enumerate(range(*idx_range)):
                    coronal_slice = image_np[:, s, :]
                    coronal_slice = scipy.ndimage.rotate(coronal_slice, 90)
                    coronal_slice = np.flip(coronal_slice, axis=1)
                    coronal_slice = scipy.ndimage.zoom(coronal_slice, [self.spacing[2], self.spacing[0]], order=1)
                    slice_list.append(coronal_slice)
                strip = np.concatenate(slice_list, axis=0)
                self._custom_imshow(axs[i], strip, title=subtitles[i], image_type=image_types[i])

        if view == 'sagittal':
            for i, image_np in enumerate(image_np_list):
                slice_list = []
                for s in range(*idx_range):
                    sagittal_slice = image_np[s, :, :]
                    sagittal_slice = scipy.ndimage.rotate(sagittal_slice, 90)
                    sagittal_slice = scipy.ndimage.zoom(sagittal_slice, [self.spacing[2], self.spacing[1]], order=1)
                    slice_list.append(sagittal_slice)
                strip = np.concatenate(slice_list, axis=0)
                self._custom_imshow(axs[i], strip, title=subtitles[i], image_type=image_types[i])

        # Display
        fig.suptitle(title, fontsize='x-large')
        plt.show()


    def grid(self, image_np, idx_range, view='axial', image_type='PET', title=''):
        array_size = image_np.shape
        phy_size = [int(array_size[i]*self.spacing[i]) for i in range(3)]
        w_phy, h_phy, d_phy = phy_size

        grid_size = (
                     5,
                     math.ceil((idx_range[1]-idx_range[0]) / 5)
                    )

        if view == 'axial':  grid_image_shape = (h_phy * grid_size[0], w_phy * grid_size[1])
        elif view == 'coronal': grid_image_shape = (d_phy * grid_size[0], w_phy* grid_size[1])
        elif view == 'sagittal': grid_image_shape = (d_phy * grid_size[0], h_phy* grid_size[1])

        grid_image = np.zeros(grid_image_shape)
        slice_list = []

        if view == 'axial':
            for s in range(*idx_range):
                axial_slice = image_np[:, :, s].T
                axial_slice = scipy.ndimage.zoom(axial_slice, [self.spacing[0], self.spacing[1]], order=1)
                slice_list.append(axial_slice)
            for gj in range(0, grid_size[1]):
                if gj != grid_size[1] - 1:
                    strip = np.concatenate(slice_list[gj*5:gj*5+5], axis=0)
                else:
                    strip = np.concatenate(slice_list[gj*5:], axis=0)
                grid_image[0:strip.shape[0], gj*w_phy : gj*w_phy+w_phy] = strip


        if view == 'coronal':
            for s in range(*idx_range):
                coronal_slice = image_np[:, s, :]
                coronal_slice = scipy.ndimage.rotate(coronal_slice, 90)
                coronal_slice = np.flip(coronal_slice, axis=1)
                coronal_slice = scipy.ndimage.zoom(coronal_slice, [self.spacing[2], self.spacing[0]], order=1)
                slice_list.append(coronal_slice)
            for gj in range(0, grid_size[1]):
                if gj != grid_size[1] - 1:
                    strip = np.concatenate(slice_list[gj*5:gj*5+5], axis=0)
                else:
                    strip = np.concatenate(slice_list[gj*5:], axis=0)
                grid_image[0:strip.shape[0], gj*w_phy : gj*w_phy+w_phy] = strip

        if view == 'sagittal':
            for s in range(*idx_range):
                sagittal_slice = image_np[s, :, :]
                sagittal_slice = scipy.ndimage.rotate(sagittal_slice, 90)
                sagittal_slice = scipy.ndimage.zoom(sagittal_slice, [self.spacing[2], self.spacing[1]], order=1)
                slice_list.append(sagittal_slice)
            for gj in range(0, grid_size[1]):
                if gj != grid_size[1] - 1:
                    strip = np.concatenate(slice_list[gj*5:gj*5+5], axis=0)
                else:
                    strip = np.concatenate(slice_list[gj*5:], axis=0)
                grid_image[0:strip.shape[0], gj*h_phy : gj*h_phy+h_phy] = strip

        # Display
        figsize = (5*400)/self.dpi, ((idx_range[1]-idx_range[0])/5*400)/self.dpi
        fig, ax = plt.subplots(figsize=figsize)
        self._custom_imshow(ax, grid_image, title=title, image_type=image_type)
        plt.show()
