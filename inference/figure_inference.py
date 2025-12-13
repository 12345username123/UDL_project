import matplotlib.pyplot as plt
import numpy as np

def plot_RSME_curves():
    inference_type = [
        'MFVI',
        'analytic_inference'
    ]


    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, figsize=(8, 6))

    x = np.arange(101) * 10

    for inference in inference_type:
        RSME = np.load(f"./results/{inference}_RMSE.npy", allow_pickle=False)
        ax.plot(x, RSME, label=inference)

    ax.set_ylim(bottom=0.3, top=1.0)
    ax.set_xlim(left=0,right=1000)
    ax.set_xticks(np.arange(0, 1001, 100))


    ax.set_xlabel("Number of acquired images")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Number of Acquired Images")

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()

    output_path = "./figures/RSME_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()