# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 25.0, (640, 480))
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)
#
#         # write the flipped frame
#         out.write(frame)
#
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()




import numpy as np
import cv2

cap = cv2.VideoCapture(0)

## some videowriter props
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 90
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')

cap.set(cv2.CAP_PROP_FPS, 25)
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
## open and set props
# out = cv2.VideoWriter_fourccoWriter()
# out.open('output.mp4', fourcc, fps, sz, True)

out = cv2.VideoWriter('output.mp4', fourcc, fps, sz)

while (True):
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()