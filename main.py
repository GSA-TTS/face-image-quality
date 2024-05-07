import os
import cv2
import json
from FaceImageQuality import FaceImageQuality

path = "imgs"
face_image_quality = FaceImageQuality()
with open("output.txt", "a") as f:
    for dir in os.listdir(path):
        subpath = os.path.join(path, dir)
        for filename in os.listdir(subpath):
            image = cv2.imread(os.path.join(subpath, filename))
            res, _, _ = face_image_quality.check_image_quality(image)
            print(f"{dir} {filename} {res}", file=f)
            print(
                json.dumps(face_image_quality.get_face_quality_params(image), indent=2),
                file=f,
            )
