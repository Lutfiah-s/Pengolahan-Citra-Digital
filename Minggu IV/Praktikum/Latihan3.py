import cv2
import numpy as np
import time

class RealTimeEnhancement:
    
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []

    def enhance_frame(self, frame, enhancement_type='adaptive'):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if enhancement_type == 'adaptive':

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'gamma':

            gamma = 1.2
            table = np.array([(i/255.0)**gamma*255 for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(gray, table)

        else:
            enhanced = gray

        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced


# ===============================
# MAIN PROGRAM
# ===============================

cap = cv2.VideoCapture(0)

enhancer = RealTimeEnhancement(target_fps=30)

while True:

    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = enhancer.enhance_frame(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced Real-Time", enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start
    delay = max(1, int(1000*(1/enhancer.target_fps - elapsed)))

cap.release()
cv2.destroyAllWindows()