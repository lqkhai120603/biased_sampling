import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate.benchmarking import create_single_passage_map, create_z_corridor_map, create_bug_trap_map
from src.samplers import GaussianValidStateSampler, BridgeTestValidStateSampler
from src.core.graph import NodeGraph
from src.core.prm import PRM2D
from src.samplers.uniform import UniformValidStateSampler

NUM_RUNS = 10  # change this one
MAP_SIZE = 20.0
WIDTHS = [2.0, 1.0, 0.5, 0.25]
SAMPLE_NUM = 800
K_NEIGHBORS = 15

MAPS = {
    "Single Passage": create_single_passage_map,
    "Z-Corridor": create_z_corridor_map,
    "Bug Trap": create_bug_trap_map,
}
SAMPLERS = ["Uniform", "Gaussian", "Bridge"]


def get_sampler(name, space_info):
    if name == "Uniform":
        return UniformValidStateSampler(space_info)
    elif name == "Gaussian":
        return GaussianValidStateSampler(space_info)
    elif name == "Bridge":
        return BridgeTestValidStateSampler(space_info)
    raise ValueError(f"Unknown sampler: {name}")


if __name__ == "__main__":
    results = []

    for map_name, map_fn in MAPS.items():
        for w in WIDTHS:
            for sampler_name in SAMPLERS:
                successes = 0
                checks_list = []
                times_list = []
                print(f"start to run setting map {map_name} | passage width {w} | sampler {sampler_name}")

                for seed in range(NUM_RUNS):
                    np.random.seed(seed)
                    env, start, goal = map_fn(width=w, size=MAP_SIZE)
                    si = env.create_space_information()
                    sampler = get_sampler(sampler_name, si)

                    prm = PRM2D(
                        si,
                        NodeGraph(),
                        start,
                        goal,
                        sampler=sampler,
                    )
                    path = prm.run(sample_num=SAMPLE_NUM, k=K_NEIGHBORS)
                    stats = prm.get_stats()
                    print_infos = {"success": False}
                    if path is not None:
                        successes += 1
                        print_infos["success"] = True
                    checks_list.append(stats["num_collision_checks"])
                    times_list.append(stats["total_time"] * 1000)
                    print_infos["num_collision_checks"] = stats["num_collision_checks"]
                    print_infos["total_time"] = stats["total_time"] * 1000
                    print(f"iter {seed + 1} : {print_infos}")

                success_rate = (successes / NUM_RUNS) * 100
                results.append({
                    "Map": map_name,
                    "Width": w,
                    "Sampler": sampler_name,
                    "Success Rate (%)": success_rate,
                    "Avg Collision Checks": np.mean(checks_list),
                    "Avg Time (ms)": np.mean(times_list),
                })
                print(f"{map_name} | w={w} | {sampler_name} -> {success_rate:.1f}%")

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n", df.to_string(index=False))

    # Plot Success Rate vs. Width for each map
    fig, axes = plt.subplots(1, len(MAPS), figsize=(14, 4), sharey=True)
    for ax, (map_name, group) in zip(axes, df.groupby("Map", sort=False)):
        for sampler_name, sub in group.groupby("Sampler"):
            ax.plot(sub["Width"], sub["Success Rate (%)"], marker="o", label=sampler_name)
        ax.set_title(map_name)
        ax.set_xlabel("Passage Width (w)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Success Rate (%)")
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig("success_rate.png", dpi=300)
    plt.show()
