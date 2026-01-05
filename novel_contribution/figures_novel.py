import numpy as np
import matplotlib.pyplot as plt

def acc_curves_novel_contribution():
    acquisition_fns = [
        'bald',
        'bald_stopper',
        'bald_deterministic'
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(101) * 10
    for acquisition_name in acquisition_fns:
        acc = np.load(f"./results/{acquisition_name}_acc.npy", allow_pickle=False)
        ax1.plot(x, acc * 100, label=acquisition_name)

    ax1.set_ylim(bottom=70)
    ax1.set_xlim(left=0, right=1000)
    ax1.set_xticks(np.arange(0, 1001, 100))
    ax1.set_yticks(np.arange(70, 101, 2))
    ax1.set_xlabel("Number of acquired images")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs Number of Acquired Images")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="lower right")

    mciters = np.load("./results/bald_stopper_mciters.npy", allow_pickle=False)
    x_mciters = np.arange(len(mciters)) * 10
    ax2.plot(x_mciters, mciters, marker='o', linewidth=1)
    ax2.set_xlabel("Number of acquired images")
    ax2.set_title("Average MC iterations on the pool set")
    ax2.set_ylim(1, 20)
    ax2.set_xlim(left=0, right=1000)
    ax2.set_xticks(np.arange(0, 1001, 100))
    ax2.set_yticks(np.arange(1, 21, 1))
    ax2.set_ylabel("Average number of MC iterations on the pool set")
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    output_path = "./figures/accuracy_curves_novel_contribution_with_mciters.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def acc_curves_novel_contribution_aleatoric_data():
    acquisition_fns = [
        'bald',
        'max_entropy',
        'max_entropy_avg_mciters',
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.arange(51) * 10
    for acquisition_name in acquisition_fns:
        # acc = np.load(f"./results/{acquisition_name}_aleatoric_acc.npy", allow_pickle=False)
        acc = np.load(f"../results/{acquisition_name}_aleatoric_acc.npy", allow_pickle=False)
        ax.plot(x, acc * 100, label=acquisition_name)

    ax.set_ylim(bottom=30)
    ax.set_xlim(left=0, right=500)
    ax.set_xticks(np.arange(0, 501, 50))
    ax.set_yticks(np.arange(25, 40, 2))
    ax.set_xlabel("Number of acquired images")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Number of Acquired Images")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="lower right")


    fig.tight_layout()
    output_path = "../figures/accuracy_curves_novel_contribution_aleatoric_data.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()