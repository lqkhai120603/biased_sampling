"""Probabilistic Roadmap Planner"""

from __future__ import annotations

from typing import List
import time

from src.core.graph import NodeGraph
from src.core.state import State


class PRM2D:
    def __init__(
        self,
        space_information,
        graph: NodeGraph,
        q_start: State,
        q_goal: State,
        sampler=None,
        motion_resolution=0.005
    ):
        if space_information is None:
            raise ValueError("SpaceInformation cannot be null.")

        if graph is None:
            raise ValueError("Graph cannot be null.")

        self.si = space_information
        self.graph = graph
        self.q_start = q_start.copy()
        self.q_goal = q_goal.copy()

        if self.q_start.dimension != 2 or self.q_goal.dimension != 2:
            raise ValueError("PRM2D requires 2D states.")

        if not self.si.is_valid(self.q_start):
            raise ValueError("Start state is invalid.")

        if not self.si.is_valid(self.q_goal):
            raise ValueError("Goal state is invalid.")

        if motion_resolution <= 0:
            raise ValueError("Motion resolution must be positive.")

        self.motion_resolution = float(motion_resolution)

        self.sampler = (
            sampler
            if sampler is not None
            else self.si.alloc_valid_state_sampler()
        )

        # Performance counters
        self.num_collision_checks = 0
        self.num_nodes_expanded = 0
        self.sampling_time = 0.0
        self.edge_connection_time = 0.0
        self.pathfinding_time = 0.0
        self.total_time = 0.0

        self.graph.add_node(self.q_start)
        self.graph.add_node(self.q_goal)

    def _reset_stats(self):
        self.num_collision_checks = 0
        self.num_nodes_expanded = 0
        self.sampling_time = 0.0
        self.edge_connection_time = 0.0
        self.pathfinding_time = 0.0
        self.total_time = 0.0

    def get_stats(self):
        return {
            "num_collision_checks": self.num_collision_checks,
            "num_nodes_expanded": self.num_nodes_expanded,
            "sampling_time": self.sampling_time,
            "edge_connection_time": self.edge_connection_time,
            "pathfinding_time": self.pathfinding_time,
            "total_time": self.total_time,
        }

    def _sample_valid_state(self):
        """Sample one valid state using the configured sampler."""
        state = State([0.0, 0.0])

        if self.sampler.sample(state):
            return state

        return None

    def _connect_neighbors(self, state: State, neighbors: List[State]):
        for neighbor in neighbors:
            self.num_collision_checks += 1
            last_valid = neighbor.copy()
            if self.si.check_motion(
                neighbor,
                state,
                last_valid_state=last_valid,
                resolution=self.motion_resolution
            ):
                self.graph.add_edge(state, neighbor)

    def run(self, sample_num, k: int = 5, max_attempts=None):
        if sample_num < 0:
            raise ValueError("Sample numbers must be non-negative")
        if k <= 0:
            raise ValueError("k must be positive")
        if max_attempts is None:
            max_attempts = max(100, sample_num * 50)

        self._reset_stats()
        total_start = time.perf_counter()
        
        try:
            added = 0
            attempts = 0
            while added < sample_num and attempts < max_attempts:
                attempts += 1
                # Sampling phase
                sampling_start = time.perf_counter()
                q_rand = self._sample_valid_state()
                self.sampling_time += time.perf_counter() - sampling_start
                if q_rand is None:
                    continue

                # Edge connection phase
                edge_start = time.perf_counter()
                neighbors = self.graph.k_nn_search(q_rand, k)
                if self.graph.add_node(q_rand):
                    self._connect_neighbors(q_rand, neighbors)
                    self.num_nodes_expanded += 1
                    added += 1
                self.edge_connection_time += time.perf_counter() - edge_start

            # Pathfinding phase
            pathfinding_start = time.perf_counter()
            path = self.graph.get_shortest_path(self.q_start, self.q_goal)
            self.pathfinding_time = time.perf_counter() - pathfinding_start
            return path
        finally:
            self.total_time = time.perf_counter() - total_start
