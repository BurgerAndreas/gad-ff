import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

thisdir = os.path.dirname(os.path.abspath(__file__))


def create_quad_tree_level(level, save_name):
    """Create a quad tree figure for the specified level with line weighting"""
    if level == 0:
        # Level 0: Empty (no divisions)
        n = 1
        level1_lines = []  # no level 1 divisions
        level2_lines = []  # no level 2 divisions
        level3_lines = []  # no level 3 divisions
    elif level == 1:
        # Level 1: 2x2 grid (4 quadrants)
        n = 2
        # level1_lines = [0, 2]  # level 1 divisions (thickest)
        level1_lines = [1]  # level 1 divisions (thickest) - center line at x=1, y=1
        level2_lines = []  # no level 2 divisions
        level3_lines = []  # no level 3 divisions
    elif level == 2:
        # Level 2: 4x4 grid (16 quadrants)
        n = 4
        level1_lines = [0, 2, 4]  # level 1 divisions (thickest)
        level2_lines = [1, 3]  # level 2 divisions (medium)
        level3_lines = []  # no level 3 divisions
    elif level == 3:
        # Level 3: 8x8 grid (64 quadrants)
        n = 8
        level1_lines = [0, 2, 4, 6, 8]  # level 1 divisions (thickest)
        level2_lines = [1, 3, 5, 7]  # level 2 divisions (medium)
        level3_lines = []  # no level 3 divisions in this case
    else:
        raise ValueError("Level must be 0, 1, 2, or 3")

    # Draw grid with line weighting
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw level 1 lines (thickest)
    for x in level1_lines:
        ax.axvline(x, color="black", linewidth=4)
        ax.axhline(x, color="black", linewidth=4)

    # Draw level 2 lines (medium)
    for x in level2_lines:
        ax.axvline(x, color="black", linewidth=2)
        ax.axhline(x, color="black", linewidth=2)

    # Draw level 3 lines (thinnest) - if any
    for x in level3_lines:
        ax.axvline(x, color="black", linewidth=1)
        ax.axhline(x, color="black", linewidth=1)

    # Formatting
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.title(f"Quad Tree Level {level}")
    plt.tight_layout(pad=0.0)

    fname = os.path.join(thisdir, save_name)
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()


# Create all four levels
create_quad_tree_level(0, "quad_tree_level0.png")
create_quad_tree_level(1, "quad_tree_level1.png")
create_quad_tree_level(2, "quad_tree_level2.png")
create_quad_tree_level(3, "quad_tree_level3.png")
