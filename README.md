# Handwritten Math Expression to LaTeX
1. Prerequisites:
    - Create a virtual environment  
      `python -m venv <your virtual environment name>`
    - Install dependencies  
      `pip install -r requirements.txt`

2. Train  
    `python3 train.py --config /config/configs.yaml`
    This loads the model configuration.  
    After training, the model will produce a `.pt` file. This file represents the model state after training and can be used as a pretrained model.  
    The `.pt` file can be loaded to initialize a model as pretrained.

3. Evaluation  
    `python3 eval.py --checkpoint /checkpoint/<model file>`
    The evaluates the performance of a model on a test set.
