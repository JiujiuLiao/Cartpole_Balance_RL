import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from libcamera import controls
import math
import time
import zmq
import argparse
from queue import Queue

# Command line option: --no-gui to run without display (pure SSH)
parser = argparse.ArgumentParser()
parser.add_argument("--no-gui", action="store_true", help="Run without cv2.imshow (for SSH)")
args = parser.parse_args()

picam = Picamera2()
config = picam.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)},
    controls = {"FrameRate": 60}
)
picam.configure(config)
picam.start()
time.sleep(1)
try:
    picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
except:
    pass

# hsv for yellow and blue
gelb_min = np.array([20, 90, 90], np.uint8)
gelb_max = np.array([30, 255, 255], np.uint8)
blau_min = np.array([100, 150, 30], np.uint8)
blau_max = np.array([120, 255, 255], np.uint8)

skalierung = 0.25

# center of rotation in the image
x_ori = int(280*skalierung)
y_ori = int(480*skalierung)

# image resolution
width = int(640*skalierung)
height = int(480*skalierung)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:9999")

# average last 5 angles and angular velocities
angles = Queue(maxsize=5)
angles.put((math.pi, time.time())) # start with 180 degrees, pendulum is hanging down
angles.put((math.pi, time.time()))
angles.put((math.pi, time.time()))
angles.put((math.pi, time.time()))
angles.put((math.pi, time.time()))

def find_largest_contour(gelb_konturen, blau_konturen):
    """
    finds the largest contour in the list of contours

    Args:
    gelb_konturen: list of yellow contours
    blau_konturen: list of blue contours

    Returns:
    color of the largest contour
    """

    # calculate areas of yellow contours
    if len(gelb_konturen) > 0:
        gelb_flaechen = [cv2.contourArea(kontur) for kontur in gelb_konturen]
        groesste_gelb_index = np.argmax(gelb_flaechen)
        groesste_gelb_flaeche = gelb_flaechen[groesste_gelb_index]
    else:
        groesste_gelb_flaeche = 0

    # calculate areas of blue contours
    if len(blau_konturen) > 0:
        blau_flaechen = [cv2.contourArea(kontur) for kontur in blau_konturen]
        groesste_blau_index = np.argmax(blau_flaechen)
        groesste_blau_flaeche = blau_flaechen[groesste_blau_index]
    else:
        groesste_blau_flaeche = 0

    # return the color of the largest contour, the largest contour
    if groesste_gelb_flaeche > groesste_blau_flaeche:
        return "gelb", gelb_konturen[groesste_gelb_index]
    else:
        return "blau", blau_konturen[groesste_blau_index]

try:
    while True:
        start = time.time()
        # read frame
        frame = picam.capture_array()

        # resize frame
        frame = cv2.resize(frame, (width, height))

        # convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create mask for yellow and blue
        gelb_maske = cv2.inRange(hsv, gelb_min, gelb_max)
        blau_maske = cv2.inRange(hsv, blau_min, blau_max)

        # find contours
        gelb_konturen, _ = cv2.findContours(gelb_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blau_konturen, _ = cv2.findContours(blau_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        for kontur in gelb_konturen:
            cv2.drawContours(frame, [kontur], -1, (0, 255, 0), 2)
        for kontur in blau_konturen:
            cv2.drawContours(frame, [kontur], -1, (255, 0, 0), 2)

        # find largest contour
        try:
            type, kontur = find_largest_contour(gelb_konturen, blau_konturen)
        except:
            continue
        M = cv2.moments(kontur)
        if M["m00"] > 0:
            # calculate center of mass
            x_medium = int((M["m10"] / M["m00"]))
            y_medium = int((M["m01"] / M["m00"]))
            cv2.rectangle(frame,(x_medium-5,y_medium-5),(x_medium+5,y_medium+5),(0,255,0),2)

        if y_medium == y_ori and x_medium > x_ori:
            if type == 'gelb':
                angle = -math.pi/2
            else:
                angle = math.pi/2
        elif y_medium == y_ori and x_medium < x_ori:
            if type == 'gelb':
                angle = math.pi/2
            else:
                angle = -math.pi/2
        elif y_medium > y_ori and x_medium == x_ori:
            if type == 'gelb':
                angle = 0
            else:
                angle = math.pi
        elif y_medium < y_ori and x_medium == x_ori:
            if type == 'gelb':
                angle = math.pi
            else:
                angle = 0
        elif y_medium < y_ori and x_medium > x_ori:
            if type == 'gelb':
                angle = -math.pi + math.atan(abs(x_medium-x_ori)/abs(y_medium-y_ori))
            else:
                angle = math.atan(abs(x_medium-x_ori)/abs(y_medium-y_ori))
        elif y_medium > y_ori and x_medium > x_ori:
            if type == 'gelb':
                angle = -math.atan(abs(x_medium-x_ori)/abs(y_ori-y_medium))
            else:
                angle = math.pi - math.atan(abs(x_medium-x_ori)/abs(y_ori-y_medium))
        elif y_medium > y_ori and x_medium < x_ori:
            if type == 'gelb':
                angle = -math.pi + math.atan(abs(x_ori-x_medium)/abs(y_ori-y_medium))
            else:
                angle = math.atan(abs(x_ori-x_medium)/abs(y_ori-y_medium))
        else:
            if type == 'gelb':
                angle = math.pi - math.atan(abs(x_ori-x_medium)/abs(y_medium-y_ori))
            else:
                angle = -math.atan(abs(x_ori-x_medium)/abs(y_medium-y_ori))

        current_time = time.time()
        time_diff = current_time - start

        if angles.full():
            angle_old, time_old = angles.get()
            angles.put((angle, current_time))
            angle_velocity_avg = (angle - angle_old) / (current_time - time_old)
        else:
            angles.put((angle, current_time))
            angle_velocity_avg = 0

        # calculate average of last 5 angles and angular velocities
        angle_avg = sum([a[0] for a in angles.queue])/len(angles.queue)

        # pole_up: 1 when pendulum is near upright (within ~5 degrees)
        pole_up = 1 if abs(angle_avg) < math.radians(5) else 0

        print(
            round(math.degrees(angles.queue[0][0]),2),
            round(math.degrees(angles.queue[1][0]),2),
            round(math.degrees(angles.queue[2][0]),2),
            round(math.degrees(angles.queue[3][0]),2),
            round(math.degrees(angles.queue[4][0]),2),
            round(angle_velocity_avg,2),
            round(time_diff,4),
            pole_up,
            round(start,2)
        )
        socket.send_string(
            str(angles.queue[0][0]) + "," +
            str(angles.queue[1][0]) + "," +
            str(angles.queue[2][0]) + "," +
            str(angles.queue[3][0]) + "," +
            str(angles.queue[4][0]) + "," +
            str(angle_velocity_avg) + "," +
            str(pole_up) + "," +
            str(start)
        )

        # draw line from center of rotation to center of mass in different colors
        if type == 'gelb':
            cv2.line(frame ,(x_ori, y_ori), (x_medium,y_medium), (255,0,255), 2)
        else:
            cv2.line(frame ,(x_ori, y_ori), (x_medium,y_medium), (255,0,0), 2)

        # show frame (only when GUI is available)
        if not args.no_gui:
            cv2.putText(frame, str(picam.capture_metadata().get('LensPosition', None)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Farbsegmentierung", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping camera...")

if not args.no_gui:
    cv2.destroyAllWindows()
picam.stop()
