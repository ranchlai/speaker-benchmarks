# Testing SOTA open-source Voxceleb speaker models on Aishell-3

## Install requirements
``` bash
pip install -r requirements.txt
```
## Run testing
``` bash
python main.py -f <aishell3-folder> -d cuda -n 32
>>> eer: 0.063

```


## Visualization

![vis](./vis.png)


## results
|  duration  | ResnetSE    | ECAPA-TDNN  |
| -- | ----------- | ----------- |
| 2s | 0.098486029 | 0.074040102 |
| 3s | 0.084023627 | 0.059502252 |
| 5s | 0.084384577 | 0.058599105 |
