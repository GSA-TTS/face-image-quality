# An Open-Source Face-Aware Capture System
Abstract: This work introduces a novel facial image capture system that utilizes computer vision technology and artificial intelligence for real-time detection and capturing of human faces. The objective of this study is to address the challenges posed by poor-quality facial images in biometric authentication, especially in passport photo acquisition and recognition. By combining face-aware capture technology with Advanced Encryption Standard (AES) encryption for secure image storage, we present a completely open-source hardware solution that consists of a Jetson processor, a 16MP autofocus RGB camera, a custom enclosure, and a touch sensor LCD for user interaction. Pilot data collection demonstrates the system's ability to capture high-quality images, achieving a 98.98% accuracy in storing images of acceptable quality. The integration of AES encryption ensures data security, making the proposed system suitable for real-time applications in other domains beyond identity verification in passport applications, such as security systems, video conferencing, etc.

<!-- ![Alt text](asset/face_aware.png) -->

# Overview
This is the modified version of original paper

Paper: https://www.mdpi.com/2079-9292/13/7/1178
github: https://github.com/baset-sarker/face-aware-capture.git 

Some functionality added here to get results, values etc.
Face quality checking thresholds are in the thresholds.py file
Error messages are in the messages.py file


# How to run the code
```console
git clone https://github.com/baset-sarker/face-image-quality.git
cd face-image-quality
pip install -r requirements.txt
```

## How to use
## To check face quality parameters
```python
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_params = face_image_quality.get_face_quality_params(image)
print(face_quality_params)
``` 

## Get face quality results only
Return results in array
['brightness', 'blur', 'background_color', 'washed_out', 'pixelation', 'face_present', 'nose_position', 'pitch', 'roll', 'eye', 'mouth']
```python 
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_results_only = face_image_quality.get_face_quality_results_only(image)
print(face_quality_results_only)
```


## Get face quality values only
Return values in array
[brightness, blur, background_color, washed_out, pixelation, face_present, nose_position, pitch, roll, eye, mouth]
The values can not be represented as a single value output None

```python
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_values_only = face_image_quality.get_face_quality_values_only(image)
print(face_quality_values_only)
```


## Check image quality pass/fail and message
```python
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
res,message = face_image_quality.check_image_quality(image)
print(res,',',message)
```


## Check image quality all images inside a folder
```python

import os
from FaceImageQuality import FaceImageQuality
path = "path of the folder"
face_image_quality = FaceImageQuality()
for filename in os.listdir(path):
    image = cv2.imread(os.path.join(path,filename))
    res,message = face_image_quality.check_image_quality(image)
    print(filename,',',res,',',message)
```

# Original version
https://github.com/baset-sarker/face-aware-capture.git

# To cite the paper
``` console

@Article{electronics13071178,
AUTHOR = {Sarker, Md Abdul Baset and Hossain, S. M. Safayet and Venkataswamy, Naveenkumar G. and Schuckers, Stephanie and Imtiaz, Masudul H.},
TITLE = {An Open-Source Face-Aware Capture System},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {7},
ARTICLE-NUMBER = {1178},
URL = {https://www.mdpi.com/2079-9292/13/7/1178},
ISSN = {2079-9292},
DOI = {10.3390/electronics13071178}
}
```






