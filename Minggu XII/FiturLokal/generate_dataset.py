import cv2
import numpy as np
import os

# =========================
# CONFIG
# =========================

dataset_path = 'dataset'

classes = ['botol', 'mug', 'buku', 'remote', 'mainan']

# =========================
# AUGMENTATION FUNCTIONS
# =========================

def rotate_image(img, angle):
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    return rotated


def scale_image(img, scale):
    h, w = img.shape[:2]

    scaled = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )

    canvas = np.ones_like(img) * 255

    sh, sw = scaled.shape[:2]

    if scale < 1:
        y = (h - sh) // 2
        x = (w - sw) // 2

        canvas[y:y+sh, x:x+sw] = scaled
        return canvas

    else:
        y = (sh - h) // 2
        x = (sw - w) // 2

        return scaled[y:y+h, x:x+w]


def change_brightness(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # ubah ke int16 agar aman saat operasi minus
    v = v.astype(np.int16)

    v = np.clip(v + value, 0, 255)

    # kembali ke uint8
    v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img

def add_occlusion(img):
    h, w = img.shape[:2]

    occ = img.copy()

    x1 = int(w * 0.3)
    y1 = int(h * 0.3)

    x2 = int(w * 0.7)
    y2 = int(h * 0.5)

    cv2.rectangle(
        occ,
        (x1, y1),
        (x2, y2),
        (0,0,0),
        -1
    )

    return occ


# =========================
# GENERATE
# =========================

for cls in classes:

    folder = os.path.join(dataset_path, cls)

    ref_path = os.path.join(folder, 'ref.jpg')

    img = cv2.imread(ref_path)

    if img is None:
        print(f'Image not found: {ref_path}')
        continue

    # ROTATION
    rot = rotate_image(img, 30)
    cv2.imwrite(os.path.join(folder, 'test_rot.jpg'), rot)

    # SCALE
    scale = scale_image(img, 0.7)
    cv2.imwrite(os.path.join(folder, 'test_scale.jpg'), scale)

    # ILLUMINATION
    illum = change_brightness(img, -80)
    cv2.imwrite(os.path.join(folder, 'test_dark.jpg'), illum)

    # OCCLUSION
    occ = add_occlusion(img)
    cv2.imwrite(os.path.join(folder, 'test_occ.jpg'), occ)

    print(f'{cls} selesai')

print('\nDataset generation completed.')