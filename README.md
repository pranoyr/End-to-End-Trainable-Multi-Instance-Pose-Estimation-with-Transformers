# End-to-End Trainable Multi-Instance Pose Estimation with Transformers

# POET
Implementation of POET for pose estimation
<!-- ![alt text](https://raw.githubusercontent.com/HHTseng/video-classification/master/fig/CRNN.png) -->

# Getting Started
### COCO Dataset
Dowload the COCO Dataset and create the folder structure as mentioned below.

```
+ data 
    + annotations   
        - 1.xml
        _ 2.xml
        .
        .
    + train2017 
        - 1.jpg
        - 2.jpg
        .
        .
```


## Train
Once you have downloaded the dataset, start training ->
```
python -m torch.distributed.launch --nproc_per_node=<number-of-gpus> --use_env main.py --coco_path ./data/ --batch_size <batch-size>
```

I trained using 2 Tesla-V100 with a batch size of 6.
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path ./data/ --batch_size 6
```

## Resume from a checkpoint
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path ./data/ --batch_size <batch-size> --resume ./snapshots/model.pth
```


<!-- ## Trained Weights till 100 epochs
Accuracy is still improving, this is not the final weights after 250 epochs.
```
cd snapshots
wget https://www.dropbox.com/s/3tvcfvynuwa9wdw/model.pth?dl=0

``` -->


## Inference
```
python inference.py
```

## To Do
Evaluation script


## References
* https://arxiv.org/pdf/2103.12115.pdf
* https://github.com/facebookresearch/detr

## License
This project is licensed under the Apache License 

