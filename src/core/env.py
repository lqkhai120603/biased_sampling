"""Environment for motion planning sampling"""

from __future__ import annotations

from typing import List

from shapely.geometry import box, Polygon, Point, LineString

from src.core.state import State
from src.core.state_space import RealVectorStateSpace
from src.samplers.space_info import SpaceInformation


class Environment2D:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid environment bounds.")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.boundary = box(self.x_min, self.y_min, self.x_max, self.y_max)
        self.obstacles: List[Polygon] = []

    def add_obstacle(self, vertices):
        obstacle = Polygon(vertices)
        if obstacle.is_empty or not obstacle.is_valid:
            raise ValueError("Obstacle must be a valid, non-empty polygon.")
        if not self.boundary.covers(obstacle):
            raise ValueError("Obstacle must lie inside the environment.")
        self.obstacles.append(obstacle)

    def is_state_valid(self, state: State):
        """Collision checker"""
        if not isinstance(state, State):
            raise TypeError("Expected a State instance.")
        if state.dimension != 2:
            raise ValueError("Environment2D requires 2D states.")

        point = Point(float(state[0]), float(state[1]))

        if not self.boundary.covers(point):
            return False

        return not any(obs.contains(point) for obs in self.obstacles)

    def is_edge_valid(self, state1, state2):
        """Check if the line between state1 and state2 is valid (collision free) or not"""
        if not isinstance(state1, State) or not isinstance(state2, State):
            raise TypeError("Expected State instances.")

        if state1.dimension != 2 or state2.dimension != 2:
            raise ValueError("Environment2D requires 2D states.")

        p1 = (float(state1[0]), float(state1[1]))
        p2 = (float(state2[0]), float(state2[1]))
        line = LineString([p1, p2])

        if not self.boundary.covers(line):
            return False

        return not any(line.intersects(obs) for obs in self.obstacles)

    def create_state_space(self):
        """Create a state space matching the environment bounds"""

        space = RealVectorStateSpace(2)
        space.set_bounds(
            [self.x_min, self.y_min],
            [self.x_max, self.y_max]
        )
        return space

    def create_space_information(self):
        """Create space information using collision checker"""

        space = self.create_state_space()
        return SpaceInformation(
            space=space,
            check_validity_fn=self.is_state_valid,
            check_motion_fn=self.is_edge_valid,
        )
