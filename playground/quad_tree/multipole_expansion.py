import matplotlib.pyplot as plt
import numpy as np
import os

thisdir = os.path.dirname(os.path.abspath(__file__))

# internal_color = "midnightblue"
# external_color = "seagreen"
# arrow_color = "#2F2F2F"

internal_color = "black"
external_color = "black"
arrow_color = "black"

head_width = 0.1
head_length = 0.15
# head_width = 0.05
# head_length = 0.08

figsize = (6, 3)

# Define points, with one interior point moved slightly inside the circle
center = (0, 0)
interior_points = [
    (0.54, 0.72),
    (-0.7, 0.3),
    (0.4, -0.5),
    (-0.2, -0.7),
    (0.9, -0.1),
    (-0.4, 0.6),
]
external_point = (3.5, 0.4)


def plot_multipole_expansion(interior_points, external_point):
    fig, ax = plt.subplots(figsize=figsize)

    # Draw circle
    circle = plt.Circle(
        center, 1.0, fill=False, linewidth=1.5, color=internal_color, zorder=1
    )
    ax.add_patch(circle)

    # Plot the center and points as dots
    ax.scatter(*center, marker="o", color="black", s=50, zorder=2)
    for pt in interior_points:
        ax.scatter(*pt, marker="o", color=internal_color, s=50, zorder=2)
    ax.scatter(*external_point, marker="o", color=external_color, s=50, zorder=2)

    # Draw arrows from interior points to center
    for x, y in interior_points:
        dx, dy = -x * 0.93, -y * 0.93
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            color=internal_color,
            zorder=3,
        )

    # Draw arrow from center to external point
    dx_ext, dy_ext = (
        (external_point[0] - center[0]) * 0.98,
        (external_point[1] - center[1]) * 0.98,
    )
    ax.arrow(
        center[0],
        center[1],
        dx_ext,
        dy_ext,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True,
        color=arrow_color,
        zorder=3,
    )

    # Formatting
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.0)
    ax.axis("off")
    # plt.show()
    fname = os.path.join(thisdir, "multipole_expansion.png")
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()


def plot_interior_to_external(interior_points, external_point):
    fig, ax = plt.subplots(figsize=figsize)

    # Draw circle
    circle = plt.Circle(
        center, 1.0, fill=False, linewidth=1.5, color=internal_color, zorder=1
    )
    ax.add_patch(circle)

    # Plot the center and points as dots
    # ax.scatter(*center, marker="o", color=internal_color, s=50, zorder=2)
    for pt in interior_points:
        ax.scatter(*pt, marker="o", color=internal_color, s=50, zorder=2)
    ax.scatter(*external_point, marker="o", color=external_color, s=50, zorder=2)

    # Draw arrows from interior points to external point
    for x, y in interior_points:
        dx, dy = (external_point[0] - x) * 0.98, (external_point[1] - y) * 0.98
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            color=arrow_color,
            zorder=3,
        )

    # Formatting
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.0)
    ax.axis("off")
    # plt.show()
    fname = os.path.join(thisdir, "interior_to_external.png")
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()


def local_expansion(interior_points, external_point):
    fig, ax = plt.subplots(figsize=figsize)

    # Draw circle
    circle = plt.Circle(
        center, 1.0, fill=False, linewidth=1.5, color=internal_color, zorder=1
    )
    ax.add_patch(circle)

    # Plot the center and points as dots
    ax.scatter(*center, marker="o", color=internal_color, s=50, zorder=2)
    for pt in interior_points:
        ax.scatter(*pt, marker="o", color=internal_color, s=50, zorder=2)
    ax.scatter(*external_point, marker="o", color=external_color, s=50, zorder=2)

    # Draw arrows from center to interior points
    for x, y in interior_points:
        dx, dy = x * 0.93, y * 0.93
        ax.arrow(
            center[0],
            center[1],
            dx,
            dy,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            color=arrow_color,
            zorder=3,
        )

    # Draw arrow from external point to center
    dx_ext, dy_ext = (
        -(external_point[0] - center[0]) * 0.98,
        -(external_point[1] - center[1]) * 0.98,
    )
    ax.arrow(
        external_point[0],
        external_point[1],
        dx_ext,
        dy_ext,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True,
        color=arrow_color,
        zorder=3,
    )

    # Formatting
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.0)
    ax.axis("off")
    # plt.show()
    fname = os.path.join(thisdir, "local_expansion.png")
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()


def external_to_internal(interior_points, external_point):
    fig, ax = plt.subplots(figsize=figsize)
    # Draw circle
    circle = plt.Circle(
        center, 1.0, fill=False, linewidth=1.5, color=internal_color, zorder=1
    )
    ax.add_patch(circle)

    # Plot the center and points as dots
    # ax.scatter(*center, marker="o", color=internal_color, s=50, zorder=2)
    for pt in interior_points:
        ax.scatter(*pt, marker="o", color=internal_color, s=50, zorder=2)
    ax.scatter(*external_point, marker="o", color=external_color, s=50, zorder=2)

    # Draw arrows from external point to interior points
    for x, y in interior_points:
        dx, dy = (x - external_point[0]) * 0.98, (y - external_point[1]) * 0.98
        ax.arrow(
            external_point[0],
            external_point[1],
            dx,
            dy,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            color=arrow_color,
            zorder=3,
        )

    # Formatting
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.0)
    ax.axis("off")
    # plt.show()
    fname = os.path.join(thisdir, "external_to_internal.png")
    plt.savefig(fname)
    print(f"Saved figure to {fname}")
    plt.close()


plot_multipole_expansion(interior_points, external_point)
plot_interior_to_external(interior_points, external_point)
local_expansion(interior_points, external_point)
external_to_internal(interior_points, external_point)
