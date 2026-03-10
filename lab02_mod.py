import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load CSV data
data = pd.read_csv('NewsHeadlinesOutput.csv')
print("=" * 60)
print(f"Data successfully loaded: {len(data)} rows")
print("=" * 60)

# Define all dimensions to analyze
dimensions = ['anger', 'fear', 'joy', 'sadness', 'sentiment']

# Define colors for each dimension
dimension_colors = {
    'anger': '#DC143C',      # red
    'fear': '#9370DB',       # purple
    'joy': '#FFD700',        # gold
    'sadness': '#4169E1',    # blue
    'sentiment': '#2E8B57'   # sea green
}

print("\nGenerating individual plots for each dimension:")
print("=" * 60)

# Store regression info for all dimensions
regression_info = {}

# First pass: calculate regression for all dimensions
for dimension in dimensions:
    human_col = dimension
    gpt_col = f'gpt{dimension}'
    dimension_data = data[[human_col, gpt_col]].dropna()
    
    if len(dimension_data) > 1:
        human_ratings = dimension_data[human_col].values
        gpt_ratings = dimension_data[gpt_col].values
        
        if len(np.unique(human_ratings)) > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    human_ratings, gpt_ratings
                )
                x_min, x_max = human_ratings.min(), human_ratings.max()
                regression_info[dimension] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_value': r_value,
                    'x_min': x_min,
                    'x_max': x_max,
                    'human_ratings': human_ratings,
                    'gpt_ratings': gpt_ratings
                }
            except Exception as e:
                print(f"✗ {dimension}: Regression failed - {e}")

# Second pass: create individual plots with background references
for focal_dimension in dimensions:
    if focal_dimension not in regression_info:
        print(f"○ {focal_dimension}: Skipped (no regression data)")
        continue
    
    # Create new figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # First, plot all OTHER dimensions as gray background lines
    for dimension in dimensions:
        if dimension == focal_dimension or dimension not in regression_info:
            continue
        
        info = regression_info[dimension]
        x_line = np.linspace(1, 7, 100)
        y_line = info['slope'] * x_line + info['intercept']
        
        # Plot thin gray line
        line, = ax.plot(
            x_line,
            y_line,
            color='#CCCCCC',
            linewidth=3,
            alpha=0.6,
            zorder=1
        )


    # Now plot the FOCAL dimension with scatter points and colored line
    focal_info = regression_info[focal_dimension]
    human_ratings = focal_info['human_ratings']
    gpt_ratings = focal_info['gpt_ratings']
    
    # Scatter plot for focal dimension
    ax.scatter(
        human_ratings,
        gpt_ratings,
        c=dimension_colors[focal_dimension],
        s=80,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.5,
        zorder=3
    )
    
    # Regression line for focal dimension
    x_line = np.linspace(1, 7, 100)
    y_line = focal_info['slope'] * x_line + focal_info['intercept']
    
    ax.plot(
        x_line,
        y_line,
        color=dimension_colors[focal_dimension],
        linewidth=6,
        alpha=0.8,
        zorder=4
    )
    
    # Confidence interval for focal dimension
    predict_error = np.sqrt(
        np.sum((gpt_ratings - (focal_info['slope'] * human_ratings + focal_info['intercept']))**2) / 
        max(len(human_ratings) - 2, 1)
    )
    
    ax.fill_between(
        x_line,
        y_line - 1.96 * predict_error,
        y_line + 1.96 * predict_error,
        color=dimension_colors[focal_dimension],
        alpha=0.15,
        zorder=2
    )
    
    # Add correlation annotation (top center)
    ax.text(
        0.5, 1.02,
        f'r = {focal_info["r_value"]:.2f}',
        fontsize=14,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Add focal dimension label
    label_x = 4.5
    label_y = focal_info['slope'] * label_x + focal_info['intercept']
    if 1 <= label_y <= 7:
        ax.text(
            label_x, label_y,
            focal_dimension,
            fontsize=18,
            fontweight='bold',
            color=dimension_colors[focal_dimension],
            ha='left',
            va='center',
            zorder=5
        )
    
    print(
        f"✓ {focal_dimension:10s}: n={len(human_ratings):3d}, "
        f"r={focal_info['r_value']:6.3f}, slope={focal_info['slope']:6.3f}"
    )
    
    # Axis labels
    ax.set_xlabel('Human Rating', fontsize=14, fontweight='bold')
    ax.set_ylabel('GPT-3.5 Turbo', fontsize=14, fontweight='bold')
    
    # Set fixed axis limits and ticks
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xticks([2, 4, 6])
    ax.set_yticks([2, 4, 6])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename_base = f'{focal_dimension}_plot'
    plt.savefig(f'{filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename_base}.pdf', bbox_inches='tight')
    
    print(f"  Saved: {filename_base}.png and {filename_base}.pdf")
    
    plt.close()

print("\n" + "=" * 60)
print("All plots generated successfully!")
print("=" * 60)

# Print detailed statistics
print("\nDetailed statistics by dimension:")
print("=" * 60)

for dimension in dimensions:
    if dimension in regression_info:
        info = regression_info[dimension]
        print(f"{dimension:10s}:")
        print(f"  - Sample size: {len(info['human_ratings'])}")
        print(f"  - Correlation: r = {info['r_value']:.3f}")
        print(f"  - Human range: [{info['human_ratings'].min():.2f}, {info['human_ratings'].max():.2f}]")
        print(f"  - GPT range: [{info['gpt_ratings'].min():.2f}, {info['gpt_ratings'].max():.2f}]")

print("=" * 60)