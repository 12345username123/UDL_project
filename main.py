from inference.figure_inference import plot_RSME_curves
from inference.main_inference import main_inference
from novel_contribution.figures_novel import acc_curves_novel_contribution, \
    acc_curves_novel_contribution_aleatoric_data
from novel_contribution.main_aleatoric import main_novel_contribution_alleatoric_data
from novel_contribution.main_bald_stopper import main_novel_contribution
from reproduction.figures_reproduction import bayesian_acc_curves, bayesian_acc_table, bayesian_vs_deterministic_curves
from reproduction.main_reproduction import main_reproduction


# This block reproduces the experiments of the paper: Run main_reproduction to compute and store the accuracy values,
# and each of the following functions plots one of the three figures and saves them into ./figures
main_reproduction()
bayesian_acc_curves()
bayesian_acc_table()
bayesian_vs_deterministic_curves()


# This block implements the minimal extension: Run main_inference to compute and store RSME values. These are then
# plotted using plot_RSME_curves. main_inference assumes that the ./features folder is filled with features extracted
# from MNIST data - to reproduce those, simple run the inference/feature_extractor.py script.
main_inference()
plot_RSME_curves()


# This block implements my novel contribution: Run main_novel_contribution/main_novel_contribution_alleatoric_data to
# compute and store accuracy data of my novel method for first and second experiment respectively.
# acc_curves_novel_contribution/acc_curves_novel_contribution_aleatoric_data then uses the created data to display results.
main_novel_contribution()
acc_curves_novel_contribution()
main_novel_contribution_alleatoric_data()
acc_curves_novel_contribution_aleatoric_data()