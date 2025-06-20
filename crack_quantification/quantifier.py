import numpy as np
import cv2
from skimage.morphology import thin
from scipy.spatial import cKDTree
from scipy.ndimage import convolve

def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    return binary

def extract_skeleton(image):
    binary = binarize(image)
    skeleton = thin(binary)
    return skeleton.astype(np.uint8)  # 0/1

def visualize_max_width(image):
    binary = binarize(image)
    skeleton = thin(binary)
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(image), 0.0

    contour_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    tree = cKDTree(contour_pts)
    max_dist, pt1, pt2 = 0, None, None

    for y, x in np.argwhere(skeleton):
        dists, idxs = tree.query([x, y], k=2)
        if len(idxs) == 2:
            p1, p2 = contour_pts[idxs[0]], contour_pts[idxs[1]]
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist, pt1, pt2 = dist, p1, p2

    output = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if pt1 is not None and pt2 is not None:
        cv2.line(output, tuple(pt1), tuple(pt2), (0, 0, 255), 2)

    return output, max_dist

def detect_branches_endpoints(image):
    binary = binarize(image)
    skeleton = thin(binary)
    skel = skeleton.astype(np.uint8)

    kernel = np.ones((3, 3), dtype=int)
    neighbors = convolve(skel, kernel, mode='constant') * skel

    endpoints = np.logical_and(skel == 1, neighbors == 2)
    branches = np.logical_and(skel == 1, neighbors >= 4)

    return int(np.sum(endpoints)), int(np.sum(branches))

def compute_features(image, pixel_size_mm, max_width_th=2.0, avg_width_th=1.0, area_ratio_th=5.0, length_th=200.0):
    binary = binarize(image)
    skeleton = thin(binary)

    area = np.sum(binary)
    length = np.sum(skeleton)
    dist_transform = cv2.distanceTransform((binary * 255).astype(np.uint8), cv2.DIST_L2, 5)
    skel_dist = dist_transform[skeleton.astype(bool)]

    avg_width = np.mean(skel_dist) * 2 if skel_dist.size > 0 else 0.0
    max_width = np.max(skel_dist) * 2 if skel_dist.size > 0 else 0.0

    kernel = np.ones((3, 3), dtype=int)
    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant') * skeleton
    endpoints = np.logical_and(skeleton == 1, neighbors == 2)
    branches = np.logical_and(skeleton == 1, neighbors >= 4)

    area_mm2 = area * (pixel_size_mm ** 2)
    length_mm = length * pixel_size_mm
    avg_width_mm = avg_width * pixel_size_mm
    max_width_mm = max_width * pixel_size_mm
    area_ratio = 100 * (area / (binary.shape[0] * binary.shape[1])) if binary.shape[0] * binary.shape[1] > 0 else 0.0

    compliance = {
        "Max Width OK": max_width_mm <= max_width_th,
        "Avg Width OK": avg_width_mm <= avg_width_th,
        "Area Ratio OK": area_ratio <= area_ratio_th,
        "Length OK": length_mm <= length_th,
    }

    vis_img, _ = visualize_max_width(image)

    return {
        "Area (mm^2)": round(area_mm2, 2),
        "Length (mm)": round(length_mm, 2),
        "Avg Width (mm)": round(avg_width_mm, 2),
        "Max Width (mm)": round(max_width_mm, 2),
        "Area Ratio (%)": round(area_ratio, 2),
        "Endpoints": int(np.sum(endpoints)),
        "Branch Points": int(np.sum(branches)),
        "Estimated Branches": max(int(np.sum(branches)) - 1, 0),
        "Pixel Size (mm)": pixel_size_mm,
        "Compliance": compliance,
        "width_visualization": vis_img
    }

def compliance_check(image, pixel_size_mm, max_width_th, avg_width_th, area_ratio_th, length_th):
    binary = binarize(image)
    total_area = binary.shape[0] * binary.shape[1]
    features = compute_features(image, pixel_size_mm)

    area_ratio = 100 * (np.sum(binary) / total_area) if total_area > 0 else 0.0

    compliance = {
        "Max Width OK": features["Max Width (mm)"] <= max_width_th,
        "Avg Width OK": features["Avg Width (mm)"] <= avg_width_th,
        "Area Ratio OK": area_ratio <= area_ratio_th,
        "Length OK": features["Length (mm)"] <= length_th
    }

    features["Area Ratio (%)"] = round(area_ratio, 2)
    features["Compliance"] = compliance
    return features
