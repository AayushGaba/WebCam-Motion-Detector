from datetime import datetime

import cv2
import time
import pandas

initial_frame = None
times = []
list_of_status = [None, None]
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if initial_frame is None:
        initial_frame = gray
        continue

    delta_frame = cv2.absdiff(initial_frame, gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (a, b, width, height) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (a, b), (a + width, b + height), (0, 255, 0), 3)

    list_of_status.append(status)
    if list_of_status[-1] == 1 and list_of_status[-2] == 0:
        times.append(datetime.now())
    if list_of_status[-1] == 0 and list_of_status[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("color Frame", frame)
    cv2.imshow("Gray frame", gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)

    pointer = cv2.waitKey(1)

    if pointer == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(list_of_status)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

df.to_csv("Times.CSV_FILE")

video.release()
cv2.destroyAllWindows()
