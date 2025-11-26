# Latex prediction from image
1. Prerequisite:
    - Create virtual enviroment
    '''python3 -m venv <your virtual enviroment name>
    - Install dependencies
    ''' pip install -r requirement.txt

2. Train
    '''python3 train..py --config /config/configs.yaml
    This will load the configuration of a model 
    After training the model will produce (save) an .pt file. This will be the model state after training perform as a pretrain
    This pt can be load and make a model become a pretrain
3. Evaluation
    '''python3 eval.py --checkpoint /checkpoint/<your .pt name>
    