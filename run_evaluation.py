import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

output_dir = './low_light_dip_outputs'
os.makedirs(output_dir, exist_ok=True)
samples_dir = os.path.join(output_dir, 'samples')
os.makedirs(samples_dir, exist_ok=True)

high_dir = './LOL_dataset/lol_dataset/eval15/high'
low_dir = './LOL_dataset/lol_dataset/eval15/low'
files = sorted(os.listdir(low_dir))

def gamma_correction(image, gamma=0.5):
    inv_gamma = gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)

def histogram_equalization(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2RGB)

def single_scale_retinex(image, sigma=80):
    img_f = image.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ret = np.log(img_f) - np.log(blur + 1e-6)
    out = np.zeros_like(ret)
    for c in range(3):
        ch = ret[:, :, c]
        c_min, c_max = ch.min(), ch.max()
        if c_max > c_min:
            out[:, :, c] = (ch - c_min) / (c_max - c_min) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def hybrid_enhancement(image, gamma=0.6, clip_limit=2.0):
    bright = gamma_correction(image, gamma)
    clahe_img = apply_clahe(bright, clip_limit=clip_limit)
    denoised = cv2.fastNlMeansDenoisingColored(clahe_img, None, 3, 3, 7, 21)
    return denoised

def calc_entropy(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def calc_contrast(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))

def calc_brightness(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))

methods = {
    'Gamma 0.4': lambda img: gamma_correction(img, 0.4),
    'Histogram Equalization': histogram_equalization,
    'CLAHE': apply_clahe,
    'Retinex sigma 80': lambda img: single_scale_retinex(img, 80),
    'Hybrid': hybrid_enhancement
}

results = []
sample_saved = False
first_sample_data = {}

for f in files:
    l_bgr = cv2.imread(os.path.join(low_dir, f))
    h_bgr = cv2.imread(os.path.join(high_dir, f))
    l_rgb = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2RGB)
    h_rgb = cv2.cvtColor(h_bgr, cv2.COLOR_BGR2RGB)

    if not sample_saved:
        first_sample_data['Original'] = l_rgb
        first_sample_data['Reference'] = h_rgb

    for name, fn in methods.items():
        t0 = time.perf_counter()
        enh = fn(l_rgb)
        t_cost = time.perf_counter() - t0
        psnr = peak_signal_noise_ratio(h_rgb, enh, data_range=255)
        ssim = structural_similarity(h_rgb, enh, channel_axis=2, data_range=255)
        entropy = calc_entropy(enh)
        contrast = calc_contrast(enh)
        brightness = calc_brightness(enh)

        results.append({
            'Image': f,
            'Method': name,
            'PSNR (dB)': round(psnr, 4),
            'SSIM': round(ssim, 4),
            'Runtime (s)': round(t_cost, 4),
            'Brightness': round(brightness, 2),
            'Contrast (StdDev)': round(contrast, 2),
            'Entropy': round(entropy, 4)
        })

        safe_name = name.replace(' ', '_').replace('=', '')
        if not sample_saved:
            first_sample_data[name] = enh
            cv2.imwrite(os.path.join(samples_dir, f'{safe_name}_{f}'), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))

    if not sample_saved:
        cv2.imwrite(os.path.join(samples_dir, f'Original_{f}'), l_bgr)
        cv2.imwrite(os.path.join(samples_dir, f'Reference_{f}'), h_bgr)
        sample_saved = True

df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, 'per_image_results.csv'), index=False)

summary_df = df.groupby('Method').agg({
    'PSNR (dB)': ['mean', 'std'],
    'SSIM': ['mean', 'std'],
    'Runtime (s)': 'mean',
    'Brightness': 'mean',
    'Contrast (StdDev)': 'mean',
    'Entropy': 'mean'
}).round(4)

summary_flat = pd.DataFrame({
    'Method': summary_df.index,
    'Mean_PSNR_dB': summary_df['PSNR (dB)']['mean'].values,
    'Std_PSNR': summary_df['PSNR (dB)']['std'].values,
    'Mean_SSIM': summary_df['SSIM']['mean'].values,
    'Std_SSIM': summary_df['SSIM']['std'].values,
    'Mean_Runtime_s': summary_df['Runtime (s)']['mean'].values,
    'Mean_Brightness': summary_df['Brightness']['mean'].values,
    'Mean_Contrast': summary_df['Contrast (StdDev)']['mean'].values,
    'Mean_Entropy': summary_df['Entropy']['mean'].values
}).sort_values('Mean_PSNR_dB', ascending=False)

summary_flat.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)
print("--- SUMMARY TABLE ---")
print(summary_flat.to_string(index=False))

# Now generate high quality figures
# 1. PSNR Bar Chart
plt.figure(figsize=(9, 5), dpi=300)
psnr_plot = summary_flat.sort_values('Mean_PSNR_dB', ascending=True)
bars = plt.barh(psnr_plot['Method'], psnr_plot['Mean_PSNR_dB'], color='#2b5c8f', edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.15, bar.get_y() + bar.get_height()/2, f'{w:.2f} dB', va='center', ha='left', fontsize=10, fontweight='bold')
plt.xlim(0, max(psnr_plot['Mean_PSNR_dB']) + 3.0)
plt.title('Low-Light Image Enhancement: Peak Signal-to-Noise Ratio (PSNR)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Mean PSNR (dB) - Higher is Better', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=300)
plt.close()

# 2. SSIM Bar Chart
plt.figure(figsize=(9, 5), dpi=300)
ssim_plot = summary_flat.sort_values('Mean_SSIM', ascending=True)
bars = plt.barh(ssim_plot['Method'], ssim_plot['Mean_SSIM'], color='#2e7d32', edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', ha='left', fontsize=10, fontweight='bold')
plt.xlim(0, max(ssim_plot['Mean_SSIM']) + 0.15)
plt.title('Low-Light Image Enhancement: Structural Similarity Index (SSIM)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Mean SSIM (0 to 1) - Higher is Better', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ssim_comparison.png'), dpi=300)
plt.close()

# 3. Runtime Bar Chart (log scale / readable)
plt.figure(figsize=(9, 5), dpi=300)
runtime_plot = summary_flat.sort_values('Mean_Runtime_s', ascending=False)
bars = plt.barh(runtime_plot['Method'], runtime_plot['Mean_Runtime_s'], color='#c62828', edgecolor='black', alpha=0.85)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.015, bar.get_y() + bar.get_height()/2, f'{w:.4f} s', va='center', ha='left', fontsize=10, fontweight='bold')
plt.xlim(0, max(runtime_plot['Mean_Runtime_s']) * 1.25)
plt.title('Computational Efficiency: Mean Execution Runtime per Image', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Execution Time (seconds) - Lower is Faster', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=300)
plt.close()

# 4. Comprehensive 6-Panel Qualitative Comparison Grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
panel_items = [
    ('Low-Light Input', first_sample_data['Original']),
    ('Gamma Correction (γ=0.4)', first_sample_data['Gamma 0.4']),
    ('Luminance Hist Eq (YCrCb)', first_sample_data['Histogram Equalization']),
    ('CLAHE (LAB L-Channel)', first_sample_data['CLAHE']),
    ('Single-Scale Retinex (σ=80)', first_sample_data['Retinex sigma 80']),
    ('Normal-Light Reference', first_sample_data['Reference'])
]

for ax, (title, img) in zip(axes.flat, panel_items):
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

plt.suptitle(f'Qualitative Visual Comparison Across Classical DIP Methods (Scene: {files[0]})', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qualitative_comparison.png'), dpi=300)
plt.close()

# 5. Histogram Comparison Plot
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
hist_items = [
    ('Original Low-Light', first_sample_data['Original']),
    ('Gamma Correction (γ=0.4)', first_sample_data['Gamma 0.4']),
    ('Luminance Hist Eq', first_sample_data['Histogram Equalization']),
    ('CLAHE', first_sample_data['CLAHE']),
    ('Retinex (σ=80)', first_sample_data['Retinex sigma 80']),
    ('Normal-Light Reference', first_sample_data['Reference'])
]

colors = ('red', 'green', 'blue')
for ax, (title, img) in zip(axes.flat, hist_items):
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, alpha=0.7, linewidth=1.5, label=col.capitalize() if ax == axes[0, 0] else "")
    ax.set_title(f'Histogram: {title}', fontsize=11, fontweight='bold')
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Intensity (0-255)', fontsize=9)
    ax.set_ylabel('Pixel Count', fontsize=9)
    ax.grid(alpha=0.3)

axes[0, 0].legend(loc='upper right')
plt.suptitle('RGB Channel Histogram Distributions: Before vs After Enhancement', fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_comparison.png'), dpi=300)
plt.close()

# 6. Metadata file
with open(os.path.join(output_dir, 'run_metadata.txt'), 'w', encoding='utf-8') as mf:
    mf.write('=====================================================\n')
    mf.write('DIGITAL IMAGE PROCESSING (CSE4007) EXPERIMENT METADATA\n')
    mf.write('=====================================================\n')
    mf.write(f"Execution Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    mf.write("Evaluation Dataset Split: LOL eval15\n")
    mf.write(f'Number of Evaluated Image Pairs: {len(files)}\n')
    mf.write('Image Dimensions: 400 x 600 x 3 (RGB uint8)\n')
    mf.write('Algorithms Tested:\n')
    mf.write('  1. Gamma Correction (gamma=0.4)\n')
    mf.write('  2. Global Luminance Histogram Equalization (YCrCb)\n')
    mf.write('  3. CLAHE on LAB L-channel (clip_limit=2.0, tile_grid=(8,8))\n')
    mf.write('  4. Single-Scale Retinex (SSR, sigma=80)\n')
    mf.write('  5. Experimental Hybrid (Gamma 0.6 + CLAHE 2.0 + FastNLMDenoising)\n')
    mf.write('Metrics Recorded: PSNR (dB), SSIM, Runtime (s), Mean Brightness, Contrast (StdDev), Shannon Entropy\n')
    mf.write('Hardware Platform: Local / CPU Execution (No GPU required)\n')
    mf.write('=====================================================\n')

print("All outputs, plots, CSVs, and metadata generated successfully in", output_dir)
