import matplotlib.pyplot as plt

# Final evaluation results
obstacles = [2, 4, 6, 8]
success_rates = [91, 84, 73, 63]

plt.figure(figsize=(8, 5))

plt.plot(
    obstacles,
    success_rates,
    marker='o',
    linewidth=2.8,
    markersize=8,
    label='Dynamic DDDQN'
)

# Value labels
for x, y in zip(obstacles, success_rates):
    plt.text(
        x,
        y + 1.5,
        f'{y}%',
        ha='center',
        fontsize=12
    )

plt.title(
    'Generalization Performance Across Dynamic Obstacle Densities',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    'Number of Dynamic Obstacles',
    fontsize=14
)

plt.ylabel(
    'Success Rate (%)',
    fontsize=14
)

plt.xticks(obstacles)
plt.ylim(50, 100)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig(
    'final_generalization_performance.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()