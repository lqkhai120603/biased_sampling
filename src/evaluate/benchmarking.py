from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon

from src.core import Environment2D, State


def visualize_prm(env, graph, q_start, q_goal, path=None, title="PRM Roadmap", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    for obs in env.obstacles:
        x, y = obs.exterior.xy
        patch = MplPolygon(np.column_stack((x, y)), closed=True, facecolor='#2c3e50', edgecolor='#1a252f', alpha=0.9,
                           zorder=2)
        ax.add_patch(patch)

    edges = list(graph.nx_graph.edges())
    if len(edges) > 0:
        lc = LineCollection(edges, colors='#95a5a6', linewidths=0.6, alpha=0.45, zorder=3, label='Roadmap Edges')
        ax.add_collection(lc)

    if len(graph.V) > 0:
        nodes = np.array([state.values for state in graph.V])
        ax.scatter(nodes[:, 0], nodes[:, 1], c='#3498db', s=12, alpha=0.8, zorder=4, label='Sampled Nodes')

    if path is not None and len(path) > 1:
        path_arr = np.array([state.values for state in path])
        ax.plot(path_arr[:, 0], path_arr[:, 1], color='#e74c3c', linewidth=3.0, zorder=5, label='Shortest Path')

    start_xy = q_start.values if hasattr(q_start, 'values') else q_start
    goal_xy = q_goal.values if hasattr(q_goal, 'values') else q_goal

    ax.scatter(start_xy[0], start_xy[1], c='#2ecc71', s=150, marker='o', edgecolors='black', linewidths=1.5, zorder=6,
               label='Start')
    ax.scatter(goal_xy[0], goal_xy[1], c='#f1c40f', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=6,
               label='Goal')

    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def create_single_passage_map(width: float, size: float = 10.0, passage_length: Optional[float] = None):
    env = Environment2D(0, size, 0, size)
    mid = 0.5 * size
    if passage_length is None:
        passage_length = 0.2 * size

    if width >= size:
        raise ValueError("width must smaller than size")

    wall_x_min = mid - (passage_length / 2.0)
    wall_x_max = mid + (passage_length / 2.0)

    gap_y_min = mid - (width / 2.0)
    gap_y_max = mid + (width / 2.0)

    if gap_y_min > 0:
        env.add_obstacle([
            (wall_x_min, 0.0),
            (wall_x_max, 0.0),
            (wall_x_max, gap_y_min),
            (wall_x_min, gap_y_min)
        ])

    if gap_y_max < size:
        env.add_obstacle([
            (wall_x_min, gap_y_max),
            (wall_x_max, gap_y_max),
            (wall_x_max, size),
            (wall_x_min, size)
        ])

    q_start = np.array([0.15 * size, mid])
    q_goal = np.array([0.85 * size, mid])
    return env, State(q_start), State(q_goal)


def create_z_corridor_map(width: float, size: float = 10.0):
    env = Environment2D(0, size, 0, size)
    p20 = 0.2 * size
    p50 = 0.5 * size
    p80 = 0.8 * size

    if width >= (p80 - p20) / 2.0:
        raise ValueError(f"width is too big compare to size")

    env.add_obstacle([
        (p20, p20 + width),
        (p50 - width / 2.0, p20 + width),
        (p50 - width / 2.0, size),
        (p20, size)
    ])

    env.add_obstacle([
        (p20, 0.0),
        (p80, 0.0),
        (p80, p20),
        (p20, p20)
    ])

    env.add_obstacle([
        (p50 + width / 2.0, p20),
        (p80, p20),
        (p80, p80 - width),
        (p50 + width / 2.0, p80 - width)
    ])

    env.add_obstacle([
        (p50 - width / 2.0, p80),
        (p80, p80),
        (p80, size),
        (p50 - width / 2.0, size)
    ])

    q_start = np.array([0.1 * size, 0.1 * size])
    q_goal = np.array([0.9 * size, 0.9 * size])
    return env, State(q_start), State(q_goal)


def create_bug_trap_map(width: float, size: float = 10.0):
    env = Environment2D(0, size, 0, size)

    wall_thick = 0.05 * size

    box_min = 0.3 * size
    box_max = 0.7 * size
    mid = 0.5 * size

    if width >= (box_max - box_min):
        raise ValueError(f"width is too big")

    env.add_obstacle([
        (box_min, box_max),
        (box_max, box_max),
        (box_max, box_max + wall_thick),
        (box_min, box_max + wall_thick)
    ])
    env.add_obstacle([
        (box_min, box_min - wall_thick),
        (box_max, box_min - wall_thick),
        (box_max, box_min),
        (box_min, box_min)
    ])

    env.add_obstacle([
        (box_max - wall_thick, box_min),
        (box_max, box_min),
        (box_max, box_max),
        (box_max - wall_thick, box_max)
    ])

    env.add_obstacle([
        (box_min, box_min),
        (box_min + wall_thick, box_min),
        (box_min + wall_thick, mid - width / 2.0),
        (box_min, mid - width / 2.0)
    ])
    env.add_obstacle([
        (box_min, mid + width / 2.0),
        (box_min + wall_thick, mid + width / 2.0),
        (box_min + wall_thick, box_max),
        (box_min, box_max)
    ])

    q_start = np.array([0.55 * size, mid])
    q_goal = np.array([0.90 * size, mid])

    return env, State(q_start), State(q_goal)


def visualize_map(env, q_start=None, q_goal=None, title="2D Environment Map", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.4)
    for obs in env.obstacles:
        x, y = obs.exterior.xy
        poly_coords = np.column_stack((x, y))
        patch = MplPolygon(poly_coords, closed=True, facecolor='#34495e', edgecolor='#1a252f', alpha=0.9)
        ax.add_patch(patch)
    if q_start is not None:
        ax.scatter(q_start[0], q_start[1], c='#2ecc71', s=140, marker='o', edgecolors='black', linewidths=1.5, zorder=5,
                   label='Start')
    if q_goal is not None:
        ax.scatter(q_goal[0], q_goal[1], c='#e74c3c', s=180, marker='*', edgecolors='black', linewidths=1.5, zorder=5,
                   label='Goal')
    if q_start is not None or q_goal is not None:
        ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
