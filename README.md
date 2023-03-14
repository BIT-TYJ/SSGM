# SSGM: Spatial Semantic Graph Matching for Loop Closure Detection in Indoor Environments

## Dataset of Scenenet 209 (Rosbag)
```sh
$ download link：https://pan.baidu.com/s/1jmmgElK0yNHfMRzDC1raYA 
$ extraction code：1il2
```
Before run the code, first change this [line](https://github.com/BIT-TYJ/SSGM/blob/c8d3cbfcfb7bab46fe2845e422aad32924c59d94/SSGM.py#L806) in SSGM.py to the location where the rosbag is saved on your computer.


## Run
```sh
$ python SSGM.py
```

## Evaluate: draw P-R curve and calculate the area of the P-R curve.
```sh
$ python analyze-scenenet-209.py
```
