import cv2
import uvc

print(uvc.device_list())

cap = uvc.Capture("1:6")

mode_id = -11
cap.frame_mode = cap.available_modes[mode_id]

while True:
    frame = cap.get_frame_robust().bgr
    print(frame.mean())
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

cap.close()
cv2.destroyAllWindows()

print("Done!")
