import numpy as np
import cv2


cap = cv2.VideoCapture("output.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps)