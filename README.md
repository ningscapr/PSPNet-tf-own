# PSPNet-tf-own

## Introduction
This is a PSPNet tensorflow implement based on my own data. The main code is based on hellochick's code and framework.
[https://github.com/hellochick/PSPNet-tensorflow](https://github.com/hellochick/PSPNet-tensorflow). I trained my own data get acceptable results.


## Train 
I used the following code to train the data. My data is 1000 pictures. I used 900 to train and 100for validation. Please change the corresponding paths when you run.
```
python3 train.py --update-mean-var --train-beta-gamma 
python3 train.py --train-beta-gamma
python3 train.py
```

The checkpoint file I get: [Checkpoint](https://drive.google.com/open?id=1PEoHw4-1G7kwPoBPPoEtQsVQM9LXuTXr).After download it, put the model_own folder in the main folder. 


## Evaluate
You could run evalutae.py to get the inferenced image. Please change the corresponding paths when you run.
```
python3 evaluate.py 
or
python3 evaluate.py --flipped-eval
```  

Both the train list and validation list are in list_own folder.


## Results

