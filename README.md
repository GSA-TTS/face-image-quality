# Overview
This is the modified version of original paper
An Open-source face-aware capture system. 

Paper: https://www.mdpi.com/2079-9292/13/7/1178
github: https://github.com/baset-sarker/face-aware-capture.git 


Some functionality added here to get results, values etc.
Face quality checking thresholds are in the thresholds.py file
Error messages are in the messages.py file
Note: Thresholds are established based on our experiment setup and may vary across different setups. Adjust the thresholds according to your specific requirements.


# How to run the code
```console
git clone https://github.com/baset-sarker/face-image-quality.git
cd face-image-quality
pip install -r requirements.txt
```

## How to use
## To check face quality parameters
Returns : {'brightness': {'result': True, 'value': 191.91988438658106, 'msg': 'The brightness of the image is within the acceptable range.'}, 'blur': {'result': True, 'value': 8.86348518232716, 'msg': 'The image is not blurry.'}.... }
```python
import cv2
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_params = face_image_quality.get_face_quality_params(image)
print(face_quality_params)
``` 

## Get face quality results only
Return results in array
['brightness', 'blur', 'background_color', 'washed_out', 'pixelation', 'face_present', 'head_position', 'pitch', 'roll', 'eye', 'mouth','red_eye']
```python 
import cv2
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_results_only = face_image_quality.get_face_quality_results_only(image)
print(face_quality_results_only)
```


## Get face quality values only
Return values in array
[brightness, blur, background_color, washed_out, pixelation, face_present, head_position, pitch, roll, eye, mouth,red_eye]
The values can not be represented as a single value output None

```python
import cv2
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
face_quality_values_only = face_image_quality.get_face_quality_values_only(image)
print(face_quality_values_only)
```


## Check image quality pass/fail and message
```python
import cv2
from FaceImageQuality import FaceImageQuality

face_image_quality = FaceImageQuality()
image = cv2.imread("image.jpg")
res,value,message = face_image_quality.check_image_quality(image)
print(res,',',value,',',message)

```


## Check image quality all images inside a folder
```python

import os
import cv2
from FaceImageQuality import FaceImageQuality

path = "path of the folder"
face_image_quality = FaceImageQuality()
for filename in os.listdir(path):
    image = cv2.imread(os.path.join(path,filename))
    res,value,message = face_image_quality.check_image_quality(image)
    print(filename,',',res,',',value,',',message)
```

# Original version
https://github.com/baset-sarker/face-aware-capture.git

# To cite the paper
```console

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






