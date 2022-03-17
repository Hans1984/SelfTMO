import numpy as np
import cv2

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def writeLDR(img, file):
    try:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
        cv2.imwrite(file, img*255.)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s"%e)

def norm(x):
    x_max = np.max(x)
    x_min = np.min(x)
    scale = x_max - x_min
    x_norm = (x - x_min)/scale
    return x_norm

def norm_mean(img):
    img = 0.5*img/img.mean()
    return img

def ulaw_np(img, scale = 10.0):
    median_value = np.median(img)
    scale = 8.759*np.power(median_value, 2.148) + 0.1494*np.power(median_value, -2.067)
    out = np.log( 1 + scale*img)/np.log(1 + scale)
    return out, scale

def load_hdr_ldr_norm_ulaw(name_hdr):

    y = cv2.imread(name_hdr, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    y_rgb = np.maximum(cv2.cvtColor(y, cv2.COLOR_BGR2RGB), 0.0)
    y_rgb = norm_mean(y_rgb)
    y_ulaw, scale = ulaw_np(y_rgb)
    return scale, y_ulaw, y_rgb
