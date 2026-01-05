# UDL Mini-Project

This repository contains the implementation for my UDL mini project. It contains a section for reproduction of some results of paper 'Deep bayesian active learning with image data', a minimal extension with a parametrised basis function setup and inference on the final linear layer, as well as a novel contribution exploring the effect of early stopping MC iterations in the AL acquisition process. It also contains scripts to display the results using matplotlib.

---

## Repository Structure

```tree
├── features/                                   # pre-computed features for parametrised basis function setups
│
├── figures/                                    # Figures displaying results created by matplotlib
│
├── inference/                                  # code used for the minimal extension
├──── active_learning_inference_pipeline.py     # complete AL training process for inference-based methods
├──── feature_extractor.py                      # training of a feature extractor, transformation and storing of MNIST using it
├──── figure_inference.py                       # creation of figure 4
├──── inference.py                              # implementation of analytical solutions of posterior and predictive
├──── main_inference.py                         # runs all minimal extension experiments if the features are already extracted
│
├── model/                                      # stored feature extractor to ensure reproducability
│
├── novel_contribution/                         # code used for the novel contribution
├──── active_learning_pipeline_mciter_count.py  # complete AL training process for novel contribution-based methods
├──── figures_novel.py                          # creation of figures 5 and 6
├──── main_aleatoric.py                         # runs experiment on modified mnist data (more aleatoric unvertainty)
├──── main_bald_stopper.py                      # runs experiment of novel method applied to BALD
├──── main_bald_stopper.py                      # runs experiment of novel method applied to BALD
├──── train_best_hyperparam_stopper.py          # find best model for an array of weight decays given current training data
│
├── reproduction/                               # code used for the reproduction
├──── active_learning_pipeline.py               # complete AL training process for reproduction
├──── figures_novel.py                          # creation of figures 1-3
├──── main_reproduction.py                      # runs all reproduction experiments
├──── train_best_hyperparam.py                  # find best model for an array of weight decays given current training data
│
├── results/                                    # stored experimental results
│
├── acquisition_fn.py                           # implementation of all used acquisition functions
├── balanced_split.py                           # helper function to create an initial 20 point dataset
├── main.py                                     # experiment runner
├── model.py                                    # used Neural network architecture and implementation of Bayesian/deterministic/novel method evaluation
│
├── README.md
```


