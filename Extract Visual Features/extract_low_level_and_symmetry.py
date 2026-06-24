
"""
This script extracts low-level and mid-level visual features from images.

Features include:
- Color statistics (HSV mean and standard deviation)
- Entropy (image complexity)
- Fractal dimension (texture complexity)
- Spatial frequency (Fourier slope and sigma)
- Edge entropy and edge density
- Symmetry (left-right and up-down)

We use some code from the Aesthetic Toolbox:
https://github.com/rbartho/aesthetics-toolbox
"""

import numpy as np
from  PIL import Image
from skimage import color
import os
import pandas as pd
from skimage.measure import shannon_entropy
from boxcounting import boxCount

### custom import
from utils.AT import fourier_modified, edge_entropy_modified, CNN_modified

def fractal_D(image: Image) -> float:
    """
    Compute the fractal dimension of a grayscale image using box-counting.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (values between 0-255).

    Returns
    -------
    float
        Estimated fractal dimension value.
    """
    # https://github.com/Phoenixfire1081/fractaldimension
    image = image/255
    image = image > 0.5
    boxCountObj = boxCount(image)
    n, r, df = boxCountObj.calculateBoxCount()
    return np.mean(df[3:8])

image_loc = r"..\Images" 
save_loc_low = r"..\Extracted Visual Features\Low"
save_loc_symmetry = r"..\Extracted Visual Features\Mid"

results_low = {"image":[], "mean_h":[], "mean_s":[], "mean_b":[], "std_h":[], "std_s":[], "std_b":[],
           "fourier_slope":[], "fourier_sigma":[], "first_order_ed":[], "second_order_ed":[],
           "edge_density":[], "entropy":[], "fractal_D":[]}

results_mid = {"image":[], "symmetry_lr":[], "symmetry_ud":[], "symmetry_lr_ud":[]}


# Load pretrained CNN kernel (AlexNet conv1), required for calculating symmetry 
[kernel,bias] = np.load(open("utils/AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)

for folder in os.listdir(image_loc):
    image_folder  = os.path.join(image_loc, folder)
    for image in os.listdir(image_folder):
        print(image)
        if image.endswith(".png"):
            img = Image.open(os.path.join(image_folder, image))
            img_gray = np.array(img.convert("L"))
            img = np.array(img)
            # Color characteristics
            img_hsv = color.rgb2hsv(img)

            results_low["image"].append(image)
            
            results_low["mean_h"].append(np.mean(img_hsv[:,:,0]))
            results_low["mean_s"].append(np.mean(img_hsv[:,:,1]))
            results_low["mean_b"].append(np.mean(img_hsv[:,:,2]))

            results_low["std_h"].append(np.std(img_hsv[:,:,0]))
            results_low["std_s"].append(np.std(img_hsv[:,:,1]))
            results_low["std_b"].append(np.std(img_hsv[:,:,2]))
            # Entropy
            results_low["entropy"].append(shannon_entropy(img_gray))
            # Fractal D
            results_low["fractal_D"].append(fractal_D(img_gray))

            sigma , slope = fourier_modified.fourier_redies(img_gray, bin_size = 2, cycles_min = 10, cycles_max=256)
            results_low["fourier_slope"].append(slope)
            results_low["fourier_sigma"].append(sigma)

            first_order, second_order, edge_d = edge_entropy_modified.first_and_second_order_entropy_and_edge_density(img_gray)
            results_low["first_order_ed"].append(first_order)
            results_low["second_order_ed"].append(second_order)
            results_low["edge_density"].append(edge_d)

            sym_lr,sym_ud, sym_lr_ud = CNN_modified.CNN_symmetry(img, kernel, bias)
            results_mid["image"].append(image)
            results_mid["symmetry_lr"].append(sym_lr)
            results_mid["symmetry_ud"].append(sym_ud)
            results_mid["symmetry_lr_ud"].append(sym_lr_ud)


results_low = pd.DataFrame.from_dict(results_low)
results_mid = pd.DataFrame.from_dict(results_mid)
# Save results
results_low.to_csv(os.path.join(save_loc_low, "low_level_features.csv"), index = False)
results_mid.to_csv(os.path.join(save_loc_symmetry, "symmetry.csv"), index = False)