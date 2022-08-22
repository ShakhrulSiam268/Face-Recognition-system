# Face Recognition System

## Face Detection

Model used : 
- [x] `Retinaface`
- [ ] `Yolov5`

## Face Recognition

Model used : `Arcface (Resnet-50)`

input size = `112x112`

Distance Type = `cosine distance` 

## Inference

Inference on test image 

`python main.py --im --path=./test/8.jpg`

Inference on webcam

`python main.py`

## Training pipeline

