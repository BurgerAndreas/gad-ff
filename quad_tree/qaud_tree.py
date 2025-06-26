import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

thisdir = os.path.dirname(os.path.abspath(__file__))

# Define grid parameters for an 8x8 quad-tree (thick every 2 cells)
n = 8
level2_lines = [0, 2, 4, 6, 8]  # thick lines
level3_lines = list(range(n + 1))  # all lines

# Original positions for 8x8 grid (before shift)
original_x_positions = (
    [
        # row 1 (top)
        (1, c)
        for c in range(3, 9)
    ]
    + [
        # row 2
        (2, 3),
        (2, 7),
        (2, 8),
    ]
    + [
        # row 3
        (3, 3),
        (3, 7),
        (3, 8),
    ]
    + [
        # row 4
        (4, 3),
        (4, 7),
        (4, 8),
    ]
    + [
        # row 5
        (5, c)
        for c in range(3, 9)
    ]
    + [
        # row 6
        (6, c)
        for c in range(3, 9)
    ]
)

# Shift x positions left by one column
x_positions = [(r, c - 1) for (r, c) in original_x_positions if 1 <= c - 1 <= n]

i_position = (3, 4)


# Draw grid
fig, ax = plt.subplots(figsize=(6, 6))

for x in level3_lines:
    lw = 1 if x not in level2_lines else 3
    ax.axvline(x, color="black", linewidth=lw)
    ax.axhline(x, color="black", linewidth=lw)

# Plot x‐marks
for r, c in x_positions:
    y = n - r
    ax.text(c + 0.5, y + 0.5, "x", va="center", ha="center", fontsize=12, zorder=4)

# Plot box i
ri, ci = i_position
y_i = n - ri
ax.text(
    ci + 0.5,
    y_i + 0.5,
    "i",
    va="center",
    ha="center",
    fontsize=12,
    fontweight="bold",
    zorder=4,
)

# Formatting
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
plt.title("Interaction list for box i")
plt.tight_layout(pad=0.0)
fname = os.path.join(thisdir, "quad_tree.png")
plt.savefig(fname)
print(f"Saved figure to {fname}")
plt.close()
