# Cerebrus Research Code

Code accompanying the research paper.

## Repository structure

- `pretrain_encoders.py`: pretrains the brain encoders.
- `encoder_models/`: candidate brain-encoder architectures.
- `encoder_classifiers/`: classifier module used during brain-encoder pretraining.
- `finetune_bla.py`: fine-tunes the Brain-Language-Action (BLA) model.
- `bla_models/`: BLA model architecture.
- `drone_sim/` and `simulation_validation.py`: simulation code used to validate the action-sequence variations generated for BLA training.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

Simulation validation requires the open-source [Webots](https://cyberbotics.com/) software to be installed before running.

BLA training uses Ray Train. To run the training code as provided, a Ray cluster must be configured and ready to use.
