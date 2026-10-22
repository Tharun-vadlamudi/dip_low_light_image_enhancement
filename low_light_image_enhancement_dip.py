"""
Low-Light Image Enhancement Using Classical Digital Image Processing Techniques
A Comparative Study of Gamma Correction, Histogram Equalization, CLAHE, and Retinex

Course: CSE4007 - Digital Image Processing
Python Implementation & Benchmarking CLI Module
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


# ==============================================================================
# ALGORITHM IMPLEMENTATIONS
# ==============================================================================

def gamma_correction(image: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """
    Applies power-law (gamma) correction to an 8-bit RGB image.
    Formula: s = c * r^gamma
    Accelerated using an 8-bit precomputed Look-Up Table (LUT).
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be strictly greater than 0, got: {gamma}")
    inv_gamma = gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization on Luminance in YCrCb color space.
    Preserves chrominance channels to prevent color artifacts.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization on L* channel of CIE-LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def single_scale_retinex(image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Single-Scale Retinex (SSR) for RGB images.
    Estimates illumination using a 2D Gaussian blur and subtracts in the log domain.
    """
    img_float = image.astype(np.float32) + 1.0
    illumination = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    retinex = np.log(img_float) - np.log(illumination + 1e-6)

    enhanced = np.zeros_like(retinex)
    for c in range(3):
        ch = retinex[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            enhanced[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255.0
        else:
            enhanced[:, :, c] = ch

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def hybrid_enhancement(image: np.ndarray, gamma: float = 0.6, clip_limit: float = 2.0) -> np.ndarray:
    """
    Proposed multi-stage hybrid enhancement pipeline:
    1. Gamma brightening (gamma=0.6)
    2. CLAHE local contrast boosting (clip_limit=2.0)
    3. Fast Non-Local Means color denoising
    """
    bright = gamma_correction(image, gamma=gamma)
    clahe_img = apply_clahe(bright, clip_limit=clip_limit, tile_grid_size=(8, 8))
    denoised = cv2.fastNlMeansDenoisingColored(
        clahe_img, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
    )
    return denoised


# ==============================================================================
# DESCRIPTIVE METRIC HELPERS
# ==============================================================================

def calc_entropy(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def calc_contrast(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def calc_brightness(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


# ==============================================================================
# PIPELINE EXECUTION & CLI
# ==============================================================================

def run_benchmark(low_dir: str, high_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    image_exts = (".png", ".jpg", ".jpeg", ".bmp")
    low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(image_exts)])
    high_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(image_exts)])
    matched = sorted(list(set(low_files) & set(high_files)))

    if not matched:
        raise ValueError(f"No matching image pairs found between {low_dir} and {high_dir}")

    print(f"Running benchmark on {len(matched)} image pairs...")

    methods = {
        "Gamma 0.4": lambda img: gamma_correction(img, 0.4),
        "Histogram Equalization": histogram_equalization,
        "CLAHE": apply_clahe,
        "Retinex sigma 80": lambda img: single_scale_retinex(img, 80),
        "Hybrid": hybrid_enhancement
    }

    all_records = []
    first_saved = False
    first_data = {}

    for idx, fn in enumerate(matched, start=1):
        p_l = os.path.join(low_dir, fn)
        p_h = os.path.join(high_dir, fn)
        bgr_l = cv2.imread(p_l)
        bgr_h = cv2.imread(p_h)

        if bgr_l is None or bgr_h is None or bgr_l.shape != bgr_h.shape:
            print(f"Skipping pair: {fn}")
            continue

        rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
        rgb_h = cv2.cvtColor(bgr_h, cv2.COLOR_BGR2RGB)

        if not first_saved:
            first_data["Original"] = rgb_l
            first_data["Reference"] = rgb_h

        for name, fn_algo in methods.items():
            t0 = time.perf_counter()
            enh = fn_algo(rgb_l)
            t_cost = time.perf_counter() - t0

            psnr = compute_psnr(rgb_h, enh, data_range=255)
            ssim = compute_ssim(rgb_h, enh, channel_axis=2, data_range=255)
            entropy = calc_entropy(enh)
            contrast = calc_contrast(enh)
            brightness = calc_brightness(enh)

            all_records.append({
                "Image": fn,
                "Method": name,
                "PSNR (dB)": round(psnr, 4),
                "SSIM": round(ssim, 4),
                "Runtime (s)": round(t_cost, 4),
                "Brightness": round(brightness, 2),
                "Contrast": round(contrast, 2),
                "Entropy": round(entropy, 4)
            })

            safe_name = name.replace(" ", "_").replace("=", "")
            if not first_saved:
                first_data[name] = enh
                cv2.imwrite(os.path.join(samples_dir, f"{safe_name}_{fn}"), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))

        if not first_saved:
            cv2.imwrite(os.path.join(samples_dir, f"Original_{fn}"), bgr_l)
            cv2.imwrite(os.path.join(samples_dir, f"Reference_{fn}"), bgr_h)
            first_saved = True

        if idx % 5 == 0 or idx == len(matched):
            print(f"Processed {idx}/{len(matched)} pairs")

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(output_dir, "per_image_results.csv"), index=False)

    summary_df = df.groupby("Method").agg({
        "PSNR (dB)": ["mean", "std"],
        "SSIM": ["mean", "std"],
        "Runtime (s)": "mean",
        "Brightness": "mean",
        "Contrast": "mean",
        "Entropy": "mean"
    }).round(4)

    summary_flat = pd.DataFrame({
        "Method": summary_df.index,
        "Mean_PSNR_dB": summary_df["PSNR (dB)"]["mean"].values,
        "Std_PSNR": summary_df["PSNR (dB)"]["std"].values,
        "Mean_SSIM": summary_df["SSIM"]["mean"].values,
        "Std_SSIM": summary_df["SSIM"]["std"].values,
        "Mean_Runtime_s": summary_df["Runtime (s)"]["mean"].values,
        "Mean_Brightness": summary_df["Brightness"]["mean"].values,
        "Mean_Contrast": summary_df["Contrast"]["mean"].values,
        "Mean_Entropy": summary_df["Entropy"]["mean"].values
    }).sort_values("Mean_PSNR_dB", ascending=False)

    summary_flat.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    print("\nBenchmark Summary Results:")
    print(summary_flat.to_string(index=False))

    return summary_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical DIP Low-Light Image Enhancement Benchmarking")
    parser.add_argument("--low_dir", type=str, default="./LOL_dataset/lol_dataset/eval15/low", help="Path to low-light images directory")
    parser.add_argument("--high_dir", type=str, default="./LOL_dataset/lol_dataset/eval15/high", help="Path to reference images directory")
    parser.add_argument("--output_dir", type=str, default="./low_light_dip_outputs", help="Output directory for CSVs and plots")
    args = parser.parse_args()

    run_benchmark(args.low_dir, args.high_dir, args.output_dir)