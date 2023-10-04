import os
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

from multiprocessing import Pool

#########

COLUMNS = [
    "elevation", 
    "aspect", 
    "aem_imag", 
    "aem_real", 
    "landsat_0", 
    "landsat_1", 
    "landsat_2", 
    "NDVI", 
    "TPI", 
    "TWI", 
    "TRI",
]

disk_path = "/Volumes/hasuriski 1/LAYERS_15"
out_folder = "combined_data"

mX = np.array([74, 0, -1, 35, 9, 23, 0, 40, -2, 50, 5])
sX = np.array([40, 70, 80, 60, 75, 65, 50, 20, 20, 35, 0.2])

def get_img_pixels(prefix, z, x, y, gray=True):
    path = f"{disk_path}/{prefix}/{z}/{x}/{y}.png"
    img = Image.open(path)
    data = np.array(img).reshape(256*256, -1)
    return data[:, 0: 1 if gray else 3]

def get_combined_pixels(z, x, y): 
    x_elevation = get_img_pixels("elev_10m", z, x, y)
    x_aspect = get_img_pixels("elev_10m_aspect", z, x, y)
    x_aem_imag = get_img_pixels("aem_imaginary_component", z, x, y)
    x_aem_real = get_img_pixels("aem_real_component", z, x, y)
    x_landsat = get_img_pixels("landsat", z, x, y, gray=False)
    x_NDVI = get_img_pixels("NDVI_max", z, x, y)
    x_TPI = get_img_pixels("TPI", z, x, y)
    x_TWI = get_img_pixels("TWI", z, x, y)
    x_TRI = get_img_pixels("TRI", z, x, y)

    return np.hstack([
        x_elevation,
        x_aspect,
        x_aem_imag,
        x_aem_real,
        x_landsat,
        x_NDVI,
        x_TPI,
        x_TWI,
        x_TRI,
    ])

def get_data(z, x, y):
    pX = get_combined_pixels(z, x, y).astype("float32")
    pX -= 128
    pX[:, 10] = np.log(256 - pX[:,10])
    pX -= mX
    pX /= sX
    pX = np.clip(pX, -5, 5)
    return pX

def foo(a):
    z, x, y = a
    f_out = f"{disk_path}/{out_folder}/{z}/{x}/{y}.npy"
    os.makedirs(os.path.dirname(f_out), exist_ok=True)
    try:
        d = get_data(z, x, y)
        # print(".", end="")
        np.save(f_out, d)
    except:
        # print(f"<{z},{x},{y}>", end="")
        return


if __name__ == "__main__":
    for z_level in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        print(z_level)

        tasks = []
        for root,_,files in os.walk(f"/Volumes/hasuriski 1/LAYERS_15/elev_10m/{z_level}"):
            for f in files:
                if not f.endswith(".png"):
                    continue
                    
                f_in = os.path.join(root, f)
                z, x, y = [int(a.replace(".png", "")) for a in f_in.split("/")[-3:]]
        
                f_out = f"{disk_path}/{out_folder}/{z}/{x}/{y}.npy"
                if os.path.isfile(f_out):
                    continue
        
                tasks.append((z, x, y))
            
        with Pool(8) as p:    
            p.map(foo, tasks)
        print()






