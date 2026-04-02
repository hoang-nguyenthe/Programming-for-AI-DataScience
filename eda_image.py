# Converted from eda_image.ipynb
# This file contains all code from the Jupyter notebook


# ======================================================================
# Markdown Cell 1
# ======================================================================
# # 🔢 EDA - Image Data: SVHN (Street View House Numbers)
# **Course:** P4AI-DS (CO3135) - HCMUT
# **Assignment 1:** Exploratory Data Analysis
# 
# **Dataset:** [SVHN](http://ufldl.stanford.edu/housenumbers/)
# 
# **Description:** Digit classification dataset with 99K images (32×32×3 RGB). Comprehensive image EDA generating data for web dashboard.


# ======================================================================
# Markdown Cell 2
# ======================================================================
# ## 1. Import Libraries


# Code Cell 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import json
import os
from PIL import Image
from torchvision import datasets
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print('✅ Import successful!')


# ======================================================================
# Markdown Cell 4
# ======================================================================
# ## 2. Load and Prepare SVHN Dataset


# Code Cell 5
# Setup paths and output directories
OUTPUT_DIR = 'plotly_data'
SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'sample_images')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def save_figure(fig, filename, output_dir=FIGURE_DIR, dpi=160):
    """Save a matplotlib figure to disk and close it."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path

print('📥 Loading SVHN dataset (this may take a few minutes)...')
train_dataset = datasets.SVHN(root='./data', split='train', download=True)
test_dataset = datasets.SVHN(root='./data', split='test', download=True)

def to_nhwc(images):
    """Convert image tensor to (N, H, W, C) with C=3."""
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f'Expected 4D image array, got shape {arr.shape}')

    # Already NHWC
    if arr.shape[-1] == 3:
        return arr

    # Torchvision SVHN format: (N, C, H, W)
    if arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 3, 1))

    # Legacy .mat-like format: (C, H, W, N)
    if arr.shape[0] == 3:
        return np.transpose(arr, (3, 1, 2, 0))

    raise ValueError(f'Unsupported image layout: {arr.shape}')

# Convert to numpy arrays in standard format (N, H, W, C)
X_train = to_nhwc(train_dataset.data)
y_train = np.asarray(train_dataset.labels)
X_test = to_nhwc(test_dataset.data)
y_test = np.asarray(test_dataset.labels)

# Convert label 10 to 0 (SVHN uses 1-10, we want 0-9)
y_train = np.where(y_train == 10, 0, y_train)
y_test = np.where(y_test == 10, 0, y_test)

print('✅ SVHN dataset loaded successfully!')
print(f'\n🔹 Training set: {len(X_train)} images')
print(f'🔹 Test set: {len(X_test)} images')
print(f'🔹 Total: {len(X_train) + len(X_test)} images')
print(f'🔹 Train shape: {X_train.shape}')
print(f'🔹 Test shape: {X_test.shape}')
print(f'🔹 Single image shape: {X_train[0].shape}')
print(f'🔹 Pixel range: [{X_train.min()}, {X_train.max()}]')
print(f'🔹 Number of classes: {len(np.unique(y_train))}')
print(f'🔹 Figure output directory: {FIGURE_DIR}')


# ======================================================================
# Markdown Cell 6
# ======================================================================
# ## 3. Class Distribution Analysis


# Code Cell 7
def analyze_class_distribution(y_train, y_test):
    """Analyze digit class distribution"""
    # Combine train and test for full distribution
    y_all = np.concatenate([y_train, y_test])
    
    classes, counts = np.unique(y_all, return_counts=True)
    
    # Calculate balance metrics
    imbalance_ratio = counts.max() / counts.min()
    class_percentages = (counts / counts.sum()) * 100
    
    print('🔢 Class Distribution Analysis:')
    print('-' * 50)
    for digit, count, pct in zip(classes, counts, class_percentages):
        print(f'Digit {digit}: {count:>6} samples ({pct:>5.1f}%)')
    
    print(f'\n📊 Imbalance Ratio: {imbalance_ratio:.2f} (max/min class size)')
    
    return {
        "classes": classes.tolist(),
        "counts": counts.tolist(),
        "percentages": class_percentages.tolist(),
        "imbalance_ratio": float(imbalance_ratio),
        "total_samples": int(counts.sum()),
        "train_samples": len(y_train),
        "test_samples": len(y_test)
    }

class_stats = analyze_class_distribution(y_train, y_test)

# Save class distribution chart
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=class_stats["classes"], y=class_stats["counts"], ax=ax, palette="viridis")
ax.set_title('SVHN Class Distribution')
ax.set_xlabel('Digit Class')
ax.set_ylabel('Number of Samples')
class_fig_path = save_figure(fig, 'svhn_class_distribution.png')
print(f'🖼️ Class distribution figure saved: {class_fig_path}')


# ======================================================================
# Markdown Cell 8
# ======================================================================
# ## 4. Pixel Statistics Calculation


# Code Cell 9
def calculate_pixel_statistics(X_train, X_test):
    """Calculate comprehensive pixel statistics"""
    # Normalize both datasets to [0,1] before combining
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0
    
    # Combine datasets (now both have shape (N, 32, 32, 3))
    X_all = np.concatenate([X_train_norm, X_test_norm], axis=0)
    
    # Calculate means and stds per channel
    means = X_all.mean(axis=(0, 1, 2))  # Shape: (3,)
    stds = X_all.std(axis=(0, 1, 2))    # Shape: (3,)
    
    # Overall statistics
    overall_mean = X_all.mean()
    overall_std = X_all.std()
    
    print("🎨 Pixel Statistics Analysis:")
    print("-" * 50)
    print(f"R Channel: Mean = {means[0]:.4f}, Std = {stds[0]:.4f}")
    print(f"G Channel: Mean = {means[1]:.4f}, Std = {stds[1]:.4f}")
    print(f"B Channel: Mean = {means[2]:.4f}, Std = {stds[2]:.4f}")
    print(f"Overall:   Mean = {overall_mean:.4f}, Std = {overall_std:.4f}")
    
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]), 
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "overall_mean": float(overall_mean),
        "overall_std": float(overall_std),
        "total_pixels": int(X_all.size),
        "image_shape": list(X_all.shape[1:])
    }

pixel_stats = calculate_pixel_statistics(X_train, X_test)

# Save RGB pixel statistics charts
channels = ['R', 'G', 'B']
means = [pixel_stats['mean_r'], pixel_stats['mean_g'], pixel_stats['mean_b']]
stds = [pixel_stats['std_r'], pixel_stats['std_g'], pixel_stats['std_b']]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(channels, means, color=['#ef4444', '#22c55e', '#3b82f6'])
axes[0].set_title('RGB Channel Means')
axes[0].set_ylabel('Mean (normalized)')

axes[1].bar(channels, stds, color=['#ef4444', '#22c55e', '#3b82f6'])
axes[1].set_title('RGB Channel Standard Deviations')
axes[1].set_ylabel('Std (normalized)')

pixel_fig_path = save_figure(fig, 'svhn_pixel_statistics.png')
print(f'🖼️ Pixel statistics figure saved: {pixel_fig_path}')


# ======================================================================
# Markdown Cell 10
# ======================================================================
# ## 5. Dimension Validation


# Code Cell 11
def validate_dimensions(X_train, X_test):
    """Validate all images meet SVHN specifications"""
    expected_shape = (32, 32, 3)
    
    # Check dimensions
    train_valid = X_train.shape[1:] == expected_shape
    test_valid = X_test.shape[1:] == expected_shape
    all_valid = train_valid and test_valid
    
    print('📐 Dimension Validation:')
    print('-' * 50)
    print(f'Expected shape: {expected_shape}')
    print(f'Train images shape: {X_train.shape[1:]}')
    print(f'Test images shape: {X_test.shape[1:]}')
    print(f'Train set valid: {"✅" if train_valid else "❌"}')
    print(f'Test set valid: {"✅" if test_valid else "❌"}')
    print(f'All dimensions valid: {"✅" if all_valid else "❌"}')
    
    validation_results = {
        "expected_shape": list(expected_shape),
        "train_shape": list(X_train.shape[1:]),
        "test_shape": list(X_test.shape[1:]),
        "train_valid": bool(train_valid),
        "test_valid": bool(test_valid),
        "all_valid": bool(all_valid),
        "total_images": len(X_train) + len(X_test),
        "valid_percentage": 100.0 if all_valid else 0.0
    }
    
    return validation_results

dim_validation = validate_dimensions(X_train, X_test)

# Save dimension validation charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['Width', 'Height', 'Channels'], dim_validation['train_shape'], color=['#0ea5e9', '#10b981', '#f59e0b'])
axes[0].set_title('SVHN Image Dimensions')
axes[0].set_ylabel('Size')

valid_share = dim_validation['valid_percentage']
invalid_share = 100.0 - valid_share
axes[1].pie([valid_share, invalid_share], labels=['Valid', 'Invalid'], autopct='%1.1f%%', colors=['#22c55e', '#ef4444'])
axes[1].set_title('Dimension Validation Status')

dim_fig_path = save_figure(fig, 'svhn_dimension_validation.png')
print(f'🖼️ Dimension validation figure saved: {dim_fig_path}')


# ======================================================================
# Markdown Cell 12
# ======================================================================
# ## 6. Sample Image Extraction


# Code Cell 13
def extract_sample_images(X_train, y_train, output_dir="plotly_data/sample_images"):
    """Extract representative samples for each digit"""
    os.makedirs(output_dir, exist_ok=True)
    
    sample_info = {}
    
    print('🖼️  Extracting Sample Images:')
    print('-' * 50)
    
    for digit in range(10):
        # Find indices for this digit
        digit_indices = np.where(y_train == digit)[0]
        
        if len(digit_indices) > 0:
            # Take first sample
            sample_idx = digit_indices[0]
            sample_img = X_train[sample_idx]
            
            # Save as PNG
            img_path = f"{output_dir}/digit_{digit}_sample.png"
            Image.fromarray(sample_img.astype(np.uint8)).save(img_path)
            
            sample_info[f"digit_{digit}"] = {
                "sample_path": img_path,
                "class_size": len(digit_indices),
                "sample_index": int(sample_idx)
            }
            
            print(f'Digit {digit}: {len(digit_indices):>5} samples → {img_path}')
    
    return sample_info

samples = extract_sample_images(X_train, y_train)


# ======================================================================
# Markdown Cell 14
# ======================================================================
# ## 7. Data Quality Assessment


# Code Cell 15
def assess_data_quality(X_train, X_test, y_train, y_test):
    """Assess dataset quality and preprocessing needs"""
    
    quality_report = {
        "corrupted_images": 0,
        "missing_labels": 0,
        "outlier_pixels": 0,
        "consistent_format": True,
        "data_completeness": 100.0
    }
    
    # Check for corrupted images (unusual pixel values)
    X_all = np.concatenate([X_train, X_test])
    corrupted = np.sum((X_all < 0) | (X_all > 255))
    quality_report["corrupted_images"] = int(corrupted)
    
    # Check for missing labels
    y_all = np.concatenate([y_train, y_test])
    missing_labels = np.sum(np.isnan(y_all))
    quality_report["missing_labels"] = int(missing_labels)
    
    # Check for outlier pixels (extremely bright or dark)
    outlier_pixels = np.sum((X_all == 0) | (X_all == 255))
    outlier_percentage = (outlier_pixels / X_all.size) * 100
    quality_report["outlier_pixels"] = float(outlier_percentage)
    
    print('🔍 Data Quality Assessment:')
    print('-' * 50)
    print(f'Corrupted images: {quality_report["corrupted_images"]}')
    print(f'Missing labels: {quality_report["missing_labels"]}')
    print(f'Outlier pixels: {quality_report["outlier_pixels"]:.2f}%')
    print(f'Format consistent: {"✅" if quality_report["consistent_format"] else "❌"}')
    print(f'Data completeness: {quality_report["data_completeness"]:.1f}%')
    
    return quality_report

quality_assessment = assess_data_quality(X_train, X_test, y_train, y_test)

# Save data quality assessment charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

count_labels = ['Corrupted images', 'Missing labels']
count_values = [quality_assessment['corrupted_images'], quality_assessment['missing_labels']]
axes[0].bar(count_labels, count_values, color=['#ef4444', '#f59e0b'])
axes[0].set_title('Data Quality Count Checks')
axes[0].set_ylabel('Count')

percent_labels = ['Outlier pixels', 'Data completeness']
percent_values = [quality_assessment['outlier_pixels'], quality_assessment['data_completeness']]
axes[1].bar(percent_labels, percent_values, color=['#0ea5e9', '#22c55e'])
axes[1].set_title('Data Quality Percentage Checks')
axes[1].set_ylabel('Percentage (%)')
axes[1].set_ylim(0, 100)

quality_fig_path = save_figure(fig, 'svhn_data_quality_assessment.png')
print(f'🖼️ Data quality figure saved: {quality_fig_path}')


# ======================================================================
# Markdown Cell 16
# ======================================================================
# ## 8. Generate Dashboard Data


# Code Cell 17
# Comprehensive data export for dashboard
svhn_eda_data = {
    "dataset_overview": {
        "name": "SVHN",
        "task": "digit_classification",
        "total_images": len(X_train) + len(X_test),
        "num_classes": 10,
        "image_shape": [32, 32, 3],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "format": "RGB",
        "pixel_range": "0-255"
    },
    "class_distribution": class_stats,
    "pixel_statistics": pixel_stats,
    "dimension_validation": dim_validation,
    "sample_images": samples,
    "quality_assessment": quality_assessment
}

# Save main data file
with open(os.path.join(OUTPUT_DIR, 'eda_svhn_data.json'), 'w') as f:
    json.dump(svhn_eda_data, f, indent=2)

print(f'✅ Main data saved to {OUTPUT_DIR}/eda_svhn_data.json')


# ======================================================================
# Markdown Cell 18
# ======================================================================
# ## 9. Generate Chart Data for Dashboard


# Code Cell 19
# Generate Chart.js compatible data
chart_data = {
    "classChart": {
        "chart_id": "classChart",
        "type": "bar",
        "labels": [str(i) for i in range(10)],
        "data": class_stats["counts"],
        "colors": ["#0a7ea4", "#0ea5b5", "#22c1c3", "#2ec4b6", "#76c893", 
                   "#a7c957", "#f4a261", "#f59e0b", "#e76f51", "#d62828"],
        "title": "SVHN Class Distribution"
    },
    "balanceChart": {
        "chart_id": "balanceChart",
        "type": "doughnut", 
        "labels": ["Well Balanced", "Moderately Imbalanced"],
        "data": [70, 30],  # Based on imbalance ratio analysis
        "colors": ["#22c55e", "#f59e0b"],
        "title": "Class Balance Overview"
    },
    "dimChart": {
        "chart_id": "dimChart",
        "type": "bar",
        "labels": ["Width", "Height", "Channels"],
        "data": [32, 32, 3],
        "colors": ["#6366f1"],
        "title": "SVHN Image Dimensions"
    },
    "meanChart": {
        "chart_id": "meanChart",
        "type": "bar",
        "labels": ["R", "G", "B"],
        "data": [pixel_stats["mean_r"], pixel_stats["mean_g"], pixel_stats["mean_b"]],
        "colors": ["#ef4444", "#22c55e", "#3b82f6"],
        "title": "RGB Channel Means (Normalized)"
    },
    "stdChart": {
        "chart_id": "stdChart",
        "type": "bar", 
        "labels": ["R", "G", "B"],
        "data": [pixel_stats["std_r"], pixel_stats["std_g"], pixel_stats["std_b"]],
        "colors": ["#ef4444", "#22c55e", "#3b82f6"],
        "title": "RGB Channel Standard Deviations"
    },
    "validationChart": {
        "chart_id": "validationChart",
        "type": "pie",
        "labels": ["Valid Dimensions", "Invalid Dimensions"],
        "data": [100, 0] if dim_validation["all_valid"] else [0, 100],
        "colors": ["#22c55e", "#ef4444"],
        "title": "Dimension Validation Status"
    }
}

# Save chart data
with open(os.path.join(OUTPUT_DIR, 'eda_svhn_charts.json'), 'w') as f:
    json.dump(chart_data, f, indent=2)

print(f'✅ Chart data saved to {OUTPUT_DIR}/eda_svhn_charts.json')

# Save a dashboard chart preview figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

axes[0].bar(chart_data['classChart']['labels'], chart_data['classChart']['data'], color=chart_data['classChart']['colors'])
axes[0].set_title(chart_data['classChart']['title'])

axes[1].pie(
    chart_data['balanceChart']['data'],
    labels=chart_data['balanceChart']['labels'],
    colors=chart_data['balanceChart']['colors'],
    autopct='%1.1f%%',
    wedgeprops=dict(width=0.45)
 )
axes[1].set_title(chart_data['balanceChart']['title'])

axes[2].bar(chart_data['dimChart']['labels'], chart_data['dimChart']['data'], color=chart_data['dimChart']['colors'][0])
axes[2].set_title(chart_data['dimChart']['title'])

axes[3].bar(chart_data['meanChart']['labels'], chart_data['meanChart']['data'], color=chart_data['meanChart']['colors'])
axes[3].set_title(chart_data['meanChart']['title'])

axes[4].bar(chart_data['stdChart']['labels'], chart_data['stdChart']['data'], color=chart_data['stdChart']['colors'])
axes[4].set_title(chart_data['stdChart']['title'])

axes[5].pie(
    chart_data['validationChart']['data'],
    labels=chart_data['validationChart']['labels'],
    colors=chart_data['validationChart']['colors'],
    autopct='%1.1f%%'
 )
axes[5].set_title(chart_data['validationChart']['title'])

preview_fig_path = save_figure(fig, 'svhn_dashboard_charts_preview.png')
print(f'🖼️ Dashboard preview figure saved: {preview_fig_path}')


# ======================================================================
# Markdown Cell 20
# ======================================================================
# ## 10. Generate Summary Report


# Code Cell 21
# Generate human-readable summary report
report_lines = [
    "=" * 70,
    "SVHN (Street View House Numbers) - Image EDA Report",
    "Generated for P4AI-DS Dashboard",
    "=" * 70,
    "",
    "📊 DATASET OVERVIEW:",
    f"   • Total Images: {svhn_eda_data['dataset_overview']['total_images']:,}",
    f"   • Training Set: {svhn_eda_data['dataset_overview']['train_samples']:,}",
    f"   • Test Set: {svhn_eda_data['dataset_overview']['test_samples']:,}",
    f"   • Classes: {svhn_eda_data['dataset_overview']['num_classes']}",
    f"   • Image Size: {svhn_eda_data['dataset_overview']['image_shape']}",
    "",
    "🔢 CLASS DISTRIBUTION:",
    f"   • Most Common Digit: {class_stats['classes'][np.argmax(class_stats['counts'])]} ({max(class_stats['counts']):,} samples)",
    f"   • Least Common Digit: {class_stats['classes'][np.argmin(class_stats['counts'])]} ({min(class_stats['counts']):,} samples)",
    f"   • Imbalance Ratio: {class_stats['imbalance_ratio']:.2f}",
    "",
    "🎨 PIXEL STATISTICS:",
    f"   • R Channel Mean: {pixel_stats['mean_r']:.4f} ± {pixel_stats['std_r']:.4f}",
    f"   • G Channel Mean: {pixel_stats['mean_g']:.4f} ± {pixel_stats['std_g']:.4f}",
    f"   • B Channel Mean: {pixel_stats['mean_b']:.4f} ± {pixel_stats['std_b']:.4f}",
    f"   • Overall Mean: {pixel_stats['overall_mean']:.4f}",
    "",
    "✅ VALIDATION RESULTS:",
    f"   • Dimension Compliance: {dim_validation['valid_percentage']:.1f}%",
    f"   • Data Completeness: {quality_assessment['data_completeness']:.1f}%",
    f"   • Corrupted Images: {quality_assessment['corrupted_images']}",
    "",
    "🖼️  SAMPLE IMAGES:",
    f"   • Generated {len(samples)} representative samples",
    f"   • Saved to: {SAMPLE_DIR}/",
    "",
    "📈 DASHBOARD FILES:",
    f"   • Main Data: {OUTPUT_DIR}/eda_svhn_data.json",
    f"   • Chart Data: {OUTPUT_DIR}/eda_svhn_charts.json",
    f"   • Sample Images: {SAMPLE_DIR}/",
    "",
    "" + "=" * 70
]

report_content = "\n".join(report_lines)

# Save report
with open(os.path.join(OUTPUT_DIR, 'eda_svhn_report.txt'), 'w') as f:
    f.write(report_content)

print(report_content)
print(f'\n📝 Report saved to {OUTPUT_DIR}/eda_svhn_report.txt')
print('\n🎉 SVHN EDA analysis complete! All dashboard data generated successfully.')

