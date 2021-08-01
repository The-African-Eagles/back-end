#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
from datetime import datetime
import pickle
from datetime import date

nnPath = str((Path(__file__).parent / Path('./models/OpenVINO_2021_2/fresh_rotten_mobilenetssdv2_openvino_2021.3_6shave.blob')).resolve().absolute())

# MobilenetSSD class labels
labelMap = ["background", "Fresh", "Rotten"]

fresh_count = 0
rotten_count = 0
data = {}
data['record'] = []
today = date.today()

# Sart defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
# DetectionNetwork class produces ImgDetections message that carries parsed
# detection results.
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(nnPath)

#nn.setConfidenceThreshold(0.7)
nn.setConfidenceThreshold(0.5)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

cam.preview.link(nn.input)

# Create XlinkOut nodes
xoutFrame = pipeline.createXLinkOut()
xoutFrame.setStreamName("rgb")
cam.preview.link(xoutFrame.input)

xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:

    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the
    # output streams defined above.
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, fr_count=[]):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            fr_count.append(labelMap[detection.label])
        cv2.imshow(name, frame)
        return fr_count

    while True:

        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        # if the frame is available, render detection data on frame and display.
        if frame is not None:
            fr_count = displayFrame("rgb", frame)
            for fr in fr_count:
                if fr == "Fresh":
                    fresh_count += 1
                elif fr == "Rotten":
                    rotten_count += 1
        try:
            frame = None
            fresh_count_p = fresh_count / (fresh_count + rotten_count) * 100
            rotten_count_p = rotten_count / (fresh_count + rotten_count) * 100
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(" We have {}% of fresh fruits and {}% of rotten fruits."
                  .format(fresh_count_p,rotten_count_p))
            #print(now)
            #print(current_time)

            #save these things in CSV and average the values then send a json
            data['record'].append({
                'date': str(today),
                'time': str(current_time),
                'Fresh_percentage': fresh_count_p,
                'Rotten_percentage': rotten_count_p
            })
            #print(data)
            with open('data.pickle', 'wb') as outfile:
                pickle.dump(data, outfile)
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            break
