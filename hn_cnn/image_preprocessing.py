import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

def resample(image, spacing, new_spacing=[1, 1], mode='constant'):
    """ Resample an image to a new spacing.
    """
    new_shape = np.round(image.shape * (np.array(spacing) / np.array(new_spacing)))
    resize_factor = new_shape / image.shape
    # new_spacing = np.array(spacing) / resize_factor
    image = ndimage.zoom(image, resize_factor, mode=mode)
    #image = cv2.resize(image, dsize=new_spacing, interpolation=cv2.INTER_LINEAR)
    return image, resize_factor

def transform_to_hu(intercept, slope, image):
    """ Transform the DICOM units to Hounsfield units
    """
    return image * slope + intercept

def get_central_coordinates(scan, crop_dim):
    """ Get the central coordinates for the tumor
    """
    # Identify the coordinates for the non-zero pixels in the slice
    x, y = np.nonzero(scan)
    xl,xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    # Calculate the coordinates to use
    xm = int((xr + xl) / 2)
    hm = int((yr + yl) / 2)
    if xm < crop_dim:
        xm = crop_dim
    if hm < crop_dim:
        hm = crop_dim
    if xm > (scan.shape[0] - crop_dim):
        xm = scan.shape[0] - crop_dim
    if hm > (scan.shape[1] - crop_dim):
        hm = scan.shape[1] - crop_dim
    return (xm, hm)

def crop_scan(scan, xm, hm, crop_dim):
    """ Crop the slice according to the central points
    """
    return scan[xm-crop_dim:xm+crop_dim, hm-crop_dim:hm+crop_dim]


def process_scan(
    scan_path,
    crop_dim=90,
    resampling=True,
    min_interval=-50,
    max_interval=300,
    gaussian_filter=True,
    spacing=[1.0, 1.0],
    mask_path=None,
):
    print(f"Processing scan {scan_path}")
    # Load the CT and mask
    scan = nib.load(scan_path)
    mask = nib.load(mask_path)
    # Check the Dicom information
    # print(img.header)
    scan_data = scan.get_fdata()
    pixels_by_slice = []
    # Select the slice to use (largest tumor area)
    for i in range(mask.shape[2]):
        scan_data_filter = np.where(
            scan_data[:, :, i] < min_interval, 0, scan_data[:, :, i]
        )
        pixels_by_slice.append(
            np.count_nonzero(np.where(scan_data_filter > max_interval, 0, scan_data_filter))
        )
    ct_slice_idx = np.argmax(pixels_by_slice)
    print(f"Chosen slice {ct_slice_idx} ({pixels_by_slice[np.argmax(pixels_by_slice)]})")
    ct_slice = scan_data[:, :, ct_slice_idx]
    mask_slice = mask.get_fdata()[:, :, ct_slice_idx]
    # Apply a gaussian filter
    if gaussian_filter:
        ct_slice = ndimage.gaussian_filter(ct_slice, sigma=0.5, order=0, truncate=3)
    ct_slice = np.where(mask_slice==0, 0, ct_slice)
    # Central coordinates based on the tumor
    (xm, hm) = get_central_coordinates(ct_slice, crop_dim)
    # Resampling the scan and the mask
    if resampling:
        original_spacing = list(scan.header.get_zooms())[0:1]
        ct_slice, factor = resample(ct_slice, original_spacing, spacing, mode='nearest')
        mask_slice, _ = resample(mask_slice/255, original_spacing, spacing)
        # Remove the noice created while performing resampling
        mask_slice = np.where(mask_slice > 0.1, 1, 0)

    ct_slice = np.where(mask_slice == 0, 0, ct_slice)
    ct_slice = crop_scan(ct_slice, int(xm*factor[0]), int(hm*factor[1]), crop_dim)
    mask_slice = crop_scan(mask_slice, int(xm*factor[0]), int(hm*factor[1]), crop_dim)

    ct_slice = ((ct_slice - min_interval)/(max_interval - min_interval)) * 255
    ct_slice = np.where(ct_slice > 255, 0, ct_slice)
    ct_slice = np.where(ct_slice <= 0, 0, ct_slice)
    ct_slice = np.where(mask_slice == 0, 0, ct_slice)
    
    # Change the orientation
    # im_cropped = np.rot90(np.flipud(im_cropped), k=1)
    # Convert to an 8-bit integer
    ct_slice = np.rint(ct_slice)
    ct_slice = np.uint8(ct_slice)
    # Store the resulting image
    return ct_slice
