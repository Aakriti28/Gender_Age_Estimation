# Bosch's Age and Gender Estimation Challenge

![Problem Cover](https://github.com/Aakriti28/Gender_Age_Estimation/blob/main/problem_cover.png)

Here is the [presentation](https://docs.google.com/presentation/d/1fVJiSLhUwnwvWKuk2bXcnlIWA-2wE9qBHJLYCCazvcs/edit?usp=sharing) made by IIT Bombay's Team at the 10th Inter IIT Tech Meet.

## Introduction
The scenes obtained from a surveillance video are usually with low resolution. Most
of the
scenes captured by a static camera are with minimal change of background. Objects
in outdoor
surveillance are often detected in far-fields. Most existing digital video surveillance
systems rely
on human observers for detecting specific activities in a real-time video scene.
However, there
are limitations in the human capability to monitor simultaneous events in surveillance
displays.
Hence, human motion analysis in automated video surveillance has become one of
the most
active and attractive research topics in the area of computer vision and pattern
recognition.

## Problem Statement
Build a solution to estimate the gender and age of people from a surveillance video
feed (like
mall, retail store, hospital etc.). Consider low resolution cameras as well as cameras
put at a
height for surveillance.

## Approach:
1. Person Tracking and Re-identificaion using DeepSort based on YOLO-v5
2. Face Detection using RetinaFace
3. Facial Super-Resolution using GFPGAN based on ESRGAN
4. Age & Gender Estimation models trained based on FaceNet as a feature
extractor

## How to run
*Note: This project needs cuda to run the code*
1. Change the working directory to the `MP_BO_T2_CODE`

2. Make sure that you fulfill all the requirements: `Python 3.8` or later with
all `requirements.txt` dependencies installed, including `torch>=1.7`. To install, run:
```
pip install -r requirements.txt 
```

3. To generate the csv files as an output for a given video:
```
python3 main.py --video 'path/../video_name.mp4' --visual 1
```
Note: `'path/../video_name.mp4'` is relative path to the video file with
`video_name` being the name of video file
`--visual` is the optional argument to save the resulting inferences with the
annotation video in the `results/` folder with the name
`video_name_infer.mp4`. (0 flag for not saving, 1 for saving the video)

4. To generate the csv files as an output for a given image:
```
python3 main_image.py --image 'path/../image_name.mp4' --visual 1
```
Note: `path/../image_name.mp4` is relative path to the video file with
image_name being the name of image file
`--visual` is the optional argument to save the resulting inferences with the
annotation video in the `results/` folder with the name
`video_name_infer.png` (0 flag for not saving, 1 for saving the video)
Running above command in terminal will generate the csv file of the inferences in
the `results/` folder with the name `video_name.csv`

Format of the output csv:
```diff
! frame num,person id, bb_xmin, bb_ymin, bb_height, bb_width, age_min, age_max, age_actual, gender 
```

## Team Members - 
* Aakriti
* Ruchir Chheda
* Valay Bundele
* Bavesh Patil
* Omkar Ghugarkar
* Harsh Kumar

Weights for pre-trained models are uploaded on [drive](https://drive.google.com/drive/folders/1syb--7KQA8QpY8G7Uu8dvsolwb64Bid4?usp=sharing). Refer to [.gitignore](.gitignore) for information about directory of storing weights.
