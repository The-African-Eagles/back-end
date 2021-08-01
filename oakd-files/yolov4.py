#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

'''
Yolo-v3 device side decoding demo
  YOLO v3 is a real-time object detection model implemented with Keras* from
  this repository <https://github.com/david8862/keras-YOLOv3-model-set> and converted
  to TensorFlow* framework. This model was pretrained on COCO* dataset with 80 classes.
'''

# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/coco2017_origin_classes.txt
labelMap = ["Fresh", "Rotten"]

syncNN = True

# Gen1 API: python3 depthai_demo.py -cnn yolo-v3 -sh 13
# pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2020_1)
yolo_v3_path = str((Path(__file__).parent / Path(
    'models/OpenVINO_2021_2/fresh_rotten_yolov4_openvino_2021.2_6shave.blob')).resolve().absolute())

if len(sys.argv) > 1:
    yolo_v3_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()
# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(416, 416)
cam.setInterleaved(False)
cam.setFps(10)

# Create a "Yolo" detection network
nn = pipeline.createYoloDetectionNetwork()


nn.setConfidenceThreshold(0.7)
nn.setNumClasses(2)
nn.setCoordinateSize(4)

# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/cfg/yolov3.cfg
# https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/yolo3_anchors.txt
# Set Yolo anchors
anchors = np.array([10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326])
nn.setAnchors(anchors)
anchorMasks52 = np.array([0,1,2])
anchorMasks26 = np.array([3,4,5])
anchorMasks13 = np.array([6,7,8])
anchorMasks = {
    "side52": anchorMasks52,
    "side26": anchorMasks26,
    "side13": anchorMasks13,
}
nn.setAnchorMasks(anchorMasks)
nn.setIouThreshold(0.5)

nn.setBlobPath(yolo_v3_path)

# Recommended configuration for best inference throughput. Compile blob file for
# have the number of available shaves and run two inference threads.
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Set Queue size to 1 to minimize latency
nn.input.setQueueSize(1)

cam.preview.link(nn.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

if(syncNN):
    nn.passthrough.link(xout_rgb.input)
else:
    cam.preview.link(xout_rgb.input)

xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("detections")
nn.out.link(xoutNN.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    frame = None
    bboxes = []

    start_time = time.time()
    counter = 0
    fps = 0
    while True:

        if(syncNN):
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None:
            bboxes = inDet.detections
            counter+=1
            current_time = time.time()
            if (current_time - start_time) > 1 :
                fps = counter / (current_time - start_time)
                counter = 0
                start_time = current_time

        if frame is not None:
            # if the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            for bbox in bboxes:
                #denormalize bounging box
                x1 = int(bbox.xmin * width)
                x2 = int(bbox.xmax * width)
                y1 = int(bbox.ymin * height)
                y2 = int(bbox.ymax * height)
                try:
                    label = labelMap[bbox.label]
                except:
                    label = bbox.label

                # Annotation data
                BOX_COLOR = (0,255,0)
                LABEL_BG_COLOR = (70, 120, 70) # greyish green background for text
                TEXT_COLOR = (255, 255, 255)   # white text
                TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

                # Set up the text for display
                cv2.rectangle(frame,(x1, y1), (x2, y1+20), LABEL_BG_COLOR, -1)
                cv2.putText(frame, label + ': %.2f' % bbox.confidence, (x1+5, y1+15), TEXT_FONT, 0.5, TEXT_COLOR, 1)
                # Set up the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, TEXT_COLOR)
            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
