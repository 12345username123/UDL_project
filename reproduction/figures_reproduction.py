import matplotlib.pyplot as plt
import numpy as np


def bayesian_acc_curves():
    acquisition_fns = [
        'max_entropy',
        'bald',
        'var_ratio',
        'mean_std',
        'random'
    ]


    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, figsize=(8, 6))

    x = np.arange(101) * 10

    for acquisition_name in acquisition_fns:
        acc = np.load(f"./results/{acquisition_name}_acc.npy", allow_pickle=False)
        ax.plot(x, acc * 100, label=acquisition_name)


    ax.set_ylim(bottom=70)
    ax.set_xlim(left=0,right=1000)
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_yticks(np.arange(70, 101, 2))


    ax.set_xlabel("Number of acquired images")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Number of Acquired Images")

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    output_path = "./figures/accuracy_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def bayesian_acc_table():
    acquisition_fns = [
        'max_entropy',
        'bald',
        'var_ratio',
        'mean_std',
        'random'
    ]

    thresholds = [0.9, 0.95]

    rows = []
    for t in thresholds:
        rows.append([
            10 * np.argmax(np.load(f"./results/{a}_acc.npy") > t)
            for a in acquisition_fns
        ])

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(
        cellText=rows,
        rowLabels=["90%", "95%"],
        colLabels=acquisition_fns,
        cellLoc="center",
        loc="center"
    )

    output_path = "./figures/accuracy_table.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def bayesian_vs_deterministic_curves():
    acquisition_fns = [
        'max_entropy',
        'bald',
        'var_ratio',
    ]

    x = np.arange(101) * 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, name in zip(axes, acquisition_fns):
        acc = np.load(f"./results/{name}_acc.npy", allow_pickle=False) * 100
        std = np.load(f"./results/{name}_std.npy", allow_pickle=False) * 100

        acc_det = np.load(f"./results/{name}_deterministic_acc.npy", allow_pickle=False) * 100
        std_det = np.load(f"./results/{name}_deterministic_std.npy", allow_pickle=False) * 100

        ax.plot(x, acc, label=f"{name}", linewidth=1.5)
        ax.fill_between(x, acc - std, acc + std, alpha=0.25)

        ax.plot(x, acc_det, label=f"{name} (deterministic)", linewidth=1.5, color='red')
        ax.fill_between(x, acc_det - std_det, acc_det + std_det, alpha=0.25, color='red')

        ax.set_xlim(0, 1000)
        ax.set_xticks(np.arange(0, 1001, 100))
        ax.set_ylim(70, 100)
        ax.set_yticks(np.arange(70, 101, 2))
        ax.set_xlabel("Number of acquired images")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize="small", loc="lower right")

    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    output_path = "./figures/bayesian_vs_deterministic.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()