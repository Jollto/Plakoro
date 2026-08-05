import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


def align_image_ecc(img, ref_img, motion_type=cv2.MOTION_TRANSLATION):
    """Aligns img to match ref_img using Enhanced Correlation Coefficient (ECC).
    Supports MOTION_TRANSLATION, MOTION_EUCLIDEAN (rigid), or MOTION_AFFINE.
    """
    # ECC requires float32 images normalized to [0, 1] for best numerical stability
    ref_float = ref_img.astype(np.float32) / 255.0
    img_float = img.astype(np.float32) / 255.0

    h, w = ref_img.shape

    # Initialize a 2x3 identity matrix for affine/translation transforms
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Termination criteria: max 200 iterations or stop when improvement drops below 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)

    try:
        # Compute the optimal transformation matrix
        _, warp_matrix = cv2.findTransformECC(
            ref_float, img_float, warp_matrix, motion_type, criteria
        )

        # Warp the original uint8 image using the calculated matrix
        aligned = cv2.warpAffine(
            img,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned
    except cv2.error as e:
        # Fallback to unaligned image if solver fails to converge on a card
        print(f"ECC alignment failed for a frame, skipping warp: {e}")
        return img


def expand_deleted_zone(alpha_channel, expand_pixels=0):
    """Takes an alpha channel, identifies deleted/transparent regions (alpha < 250),

    and expands that deleted zone outward by expand_pixels.
    """
    # 1. Create a binary mask where 255 = deleted/transparent region
    # Using 250 instead of 0 to capture semi-transparent soft edges from GIMP
    deleted_mask = (alpha_channel < 250).astype(np.uint8) * 255

    # 2. Define a structuring element (kernel) based on the expansion radius
    kernel_size = expand_pixels * 2 + 1  # 2 pixels radius -> 5x5 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 3. Dilate the mask to push the "deleted" boundaries outward by expand_pixels
    expanded_deleted_mask = cv2.dilate(deleted_mask, kernel, iterations=1)

    # Return boolean array where True = pixels to turn into NaN
    return expanded_deleted_mask > 0


# Load filenames
filenames = glob.glob("plakoro/**/*.webp")
if not filenames:
    raise FileNotFoundError("No webp files found.")

processed_stack = []
ref_image = None

for filename in filenames:
    # Read image including the Alpha channel
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Couldn't load {filename}")
        continue

    # Process 4-channel BGRA images
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        aligned_img = align_image_ecc(
            gray, ref_image, motion_type=cv2.MOTION_TRANSLATION
        )

        # Expand the NaN zone by 2 pixels
        nan_mask = expand_deleted_zone(alpha, expand_pixels=2)

        # Apply NaN to the expanded region
        gray[nan_mask] = np.nan

    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    """
    if ref_image is None:
        ref_image = gray

    aligned_img = align_image_ecc(gray, ref_image, motion_type=cv2.MOTION_TRANSLATION)
    """

    processed_stack.append(aligned_img)

# Compute median ignoring NaN values
images_array = np.array(processed_stack)
background_float = np.nanmedian(images_array, axis=0)

# Convert back to image format
background = np.nan_to_num(background_float, nan=0).astype(np.uint8)

plt.imshow(background, cmap="gray")
plt.axis("off")
plt.show()

cv2.imwrite("aligned_background_0px.png", background)
