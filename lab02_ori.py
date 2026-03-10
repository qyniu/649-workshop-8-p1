import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load CSV data
data = pd.read_csv('NewsHeadlinesOutput.csv')
print("=" * 60)
print(f"Data successfully loaded: {len(data)} rows")
print("=" * 60)

# Define all dimensions to analyze (emotions + sentiment)
dimensions = ['anger', 'fear', 'joy', 'sadness', 'sentiment']

# Define colors for each dimension
dimension_colors = {
    'anger': '#DC143C',      # red
    'fear': '#9370DB',       # purple
    'joy': '#FFD700',        # gold
    'sadness': '#4169E1',    # blue
    'sentiment': '#2E8B57'   # sea green
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))

print("\nPlotting scatter points and regression lines for each dimension:")
print("=" * 60)

# Plot scatter and regression for each dimension
for dimension in dimensions:
    human_col = dimension
    gpt_col = f'gpt{dimension}'
    
    # Extract data and remove NaN values
    dimension_data = data[[human_col, gpt_col]].dropna()
    
    if len(dimension_data) > 0:
        human_ratings = dimension_data[human_col].values
        gpt_ratings = dimension_data[gpt_col].values
        
        # Scatter plot
        ax.scatter(
            human_ratings,
            gpt_ratings,
            c=dimension_colors[dimension],
            label=dimension,
            s=80,
            alpha=0.6,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Linear regression
        if len(human_ratings) > 1 and len(np.unique(human_ratings)) > 1:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    human_ratings, gpt_ratings
                )
                
                # Regression line
                x_min, x_max = human_ratings.min(), human_ratings.max()
                x_line = np.linspace(x_min, x_max, 100)
                y_line = slope * x_line + intercept
                
                ax.plot(
                    x_line,
                    y_line,
                    color=dimension_colors[dimension],
                    linewidth=6,
                    alpha=0.8
                )
                
                # Confidence interval (approx. 95%)
                predict_error = np.sqrt(
                    np.sum((gpt_ratings - (slope * human_ratings + intercept))**2) / 
                    max(len(human_ratings) - 2, 1)
                )
                
                ax.fill_between(
                    x_line,
                    y_line - 1.96 * predict_error,
                    y_line + 1.96 * predict_error,
                    color=dimension_colors[dimension],
                    alpha=0.15
                )
                
                print(
                    f"✓ {dimension:10s}: n={len(dimension_data):3d}, "
                    f"r={r_value:6.3f}, slope={slope:6.3f}"
                )
                
            except Exception as e:
                print(f"✗ {dimension:10s}: Regression failed - {e}")
        else:
            print(f"○ {dimension:10s}: Insufficient data (n={len(dimension_data)})")

# Compute overall correlation across all dimensions
all_human = []
all_gpt = []

for dimension in dimensions:
    human_col = dimension
    gpt_col = f'gpt{dimension}'
    dimension_data = data[[human_col, gpt_col]].dropna()
    all_human.extend(dimension_data[human_col].values)
    all_gpt.extend(dimension_data[gpt_col].values)

overall_corr = np.corrcoef(all_human, all_gpt)[0, 1]

# Add correlation annotation (top center)
ax.text(
    0.5, 1.02,
    f'mean r = {overall_corr:.2f}',
    fontsize=14,
    ha='center',
    va='bottom',
    transform=ax.transAxes,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
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

# Legend - hidden
# ax.legend(loc='upper left', framealpha=0.9, fontsize=11)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)

plt.tight_layout()

# Save figure
plt.savefig('emotion_sentiment_plot.png', dpi=300, bbox_inches='tight')

# Show figure
plt.show()

# Print detailed statistics
print("\n" + "=" * 60)
print("Detailed statistics by dimension:")
print("=" * 60)

for dimension in dimensions:
    human_col = dimension
    gpt_col = f'gpt{dimension}'
    dimension_data = data[[human_col, gpt_col]].dropna()
    
    if len(dimension_data) > 1:
        human_vals = dimension_data[human_col].values
        gpt_vals = dimension_data[gpt_col].values
        corr = np.corrcoef(human_vals, gpt_vals)[0, 1]
        
        print(f"{dimension:10s}:")
        print(f"  - Sample size: {len(dimension_data)}")
        print(f"  - Correlation: r = {corr:.3f}")
        print(f"  - Human range: [{human_vals.min():.2f}, {human_vals.max():.2f}]")
        print(f"  - GPT range: [{gpt_vals.min():.2f}, {gpt_vals.max():.2f}]")

print(f"\nOverall correlation: r = {overall_corr:.3f}")
print(f"Total number of data points: {len(all_human)}")
print("\nFigure saved as: emotion_sentiment_plot.png")
print("=" * 60)