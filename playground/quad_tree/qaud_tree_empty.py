import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

thisdir = os.path.dirname(os.path.abspath(__file__))

def draw_quad_tree_empty(n, draw_i=False):
    # Define grid parameters for an 8x8 quad-tree (thick every 2 cells)
    n = 8
    level2_lines = [0, 2, 4, 6, 8]  # thick lines
    level3_lines = list(range(n + 1))  # all lines

    # Draw grid
    fig, ax = plt.subplots(figsize=(6, 6))

    for x in level3_lines:
        lw = 1 if x not in level2_lines else 3
        ax.axvline(x, color="black", linewidth=lw)
        ax.axhline(x, color="black", linewidth=lw)

    if draw_i:
        ax.text(3, 4, "i", va="center", ha="center", fontsize=12, zorder=4)

    # Formatting
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    # plt.title("Interaction list for box i")
    plt.tight_layout(pad=0.0)
    fname = os.path.join(thisdir, f"quad_tree_empty{'_i' if draw_i else ''}.png")
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()

if __name__ == "__main__":
    draw_quad_tree_empty(8, draw_i=False)
    draw_quad_tree_empty(8, draw_i=True)