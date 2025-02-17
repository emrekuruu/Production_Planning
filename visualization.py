import matplotlib.pyplot as plt 
import seaborn as sns

def line_graph(results_df):

    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_context("talk", font_scale=1.2)

    algorithms = results_df['Algorithm'].drop_duplicates().tolist()
    palette = sns.color_palette("husl", len(algorithms)) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sns.lineplot(
        data=results_df, x='Number of Parts', y='Objective Value', 
        hue='Algorithm', marker='o', ax=ax1, legend=False
    )

    ax1.set_title("Objective Values vs Number of Parts", fontsize=16, pad=15)
    ax1.set_xlabel("Number of Parts", fontsize=14)
    ax1.set_ylabel("Objective Value", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    sns.lineplot(
        data=results_df, x='Number of Parts', y='Time', 
        hue='Algorithm', marker='o', ax=ax2, legend=False
    )

    ax2.set_title("Computation Time vs Number of Parts", fontsize=16, pad=15)
    ax2.set_xlabel("Number of Parts", fontsize=14)
    ax2.set_ylabel("Time (s)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)


    custom_lines = [
        plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8) 
        for color in palette
    ]

    fig.legend(
        custom_lines, algorithms, title="Algorithm", loc='upper center', bbox_to_anchor=(0.5, 1.12),
        ncol=6, fontsize=12, title_fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def bar_graph(results_df):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_context("talk", font_scale=1.2)
    algorithms = results_df['Algorithm'].drop_duplicates().tolist()
    palette = sns.color_palette("husl", len(algorithms)) 
    sns.set_palette(palette)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Get unique values in their original order
    part_numbers = results_df['Number of Parts'].unique()

    sns.barplot(
        data=results_df, x='Number of Parts', y='Objective Value', 
        hue='Algorithm', ax=ax1, legend=False,
        order=part_numbers
    )

    ax1.set_title("Objective Values vs Number of Parts", fontsize=16, pad=15)
    ax1.set_xlabel("Number of Parts", fontsize=14)
    ax1.set_ylabel("Objective Value", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    sns.barplot(
        data=results_df, x='Number of Parts', y='Time', 
        hue='Algorithm', ax=ax2, legend=False,
        order=part_numbers
    )

    ax2.set_title("Computation Time vs Number of Parts", fontsize=16, pad=15)
    ax2.set_xlabel("Number of Parts", fontsize=14)
    ax2.set_ylabel("Time (s)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    algorithms = results_df['Algorithm'].drop_duplicates().tolist()
    custom_patches = [plt.Rectangle((0,0),1,1, facecolor=color) for color in palette]
    
    fig.legend(
        custom_patches, algorithms, title="Algorithm", loc='upper center', 
        bbox_to_anchor=(0.5, 1.12), ncol=6, fontsize=12, title_fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def parameter_line_graph(results_df):    
    unique_varieties = sorted(results_df['Number of Parts'].unique())
    num_subplots = len(unique_varieties)
    
    cols = 5
    rows = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(28, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    
    for idx, variety in enumerate(unique_varieties):
        if idx < len(axes):  # Only plot if we have an axis available
            data = results_df[results_df['Number of Parts'] == variety]
            
            sns.lineplot(
                data=data, 
                x='Number of Colors', 
                y='Time',
                marker='o',
                ax=axes[idx]
            )
            
            axes[idx].grid(True, linestyle='--', alpha=0.6)
            axes[idx].set_title(f'Number of Parts: {variety}', pad=10)
    
    # Hide empty subplots
    for idx in range(num_subplots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Optimization Time vs Number of Colors by Number of Parts', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def box_plot(results_df):
    sns.set_context("talk", font_scale=1.2)

    plt.figure(figsize=(14, 8))

    # Get unique values in their original order
    part_numbers = sorted(results_df['Number of Parts'].unique())
    
    # Create more aesthetic box plot with white boxes
    sns.boxplot(
        data=results_df, 
        x='Number of Parts', 
        y='Time',
        order=part_numbers,
        palette="husl",
        width=0.6,
        medianprops={"color": "red"},
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 6},
        boxprops={"alpha": 0.8, "facecolor": "white"}
    )

    plt.title("Computation Time Distribution by Number of Parts", fontsize=16, pad=15)
    plt.xlabel("Number of Parts", fontsize=14)
    plt.ylabel("Time (s)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()