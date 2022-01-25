import cv2


# cap = cv2.VideoCapture('rtsp://admin:l2dtech123@192.168.2.33:554/cam/realmonitor?channel=1&subtype=0')

# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setVideoSize(1920, 1080)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(100)

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
	dai.CameraControl.setManualFocus(device, 1)
	video = device.getOutputQueue(name="video", maxSize=500, blocking=False)

	while True:
		videoIn = video.get()
		# cv2.imshow("video", videoIn.getCvFrame())
		# cv2.waitKey(1)

		# Get BGR frame from NV12 encoded video frame to show with opencv
		# Visualizing the frame on slower hosts might have overhead
		im = videoIn.getCvFrame()
		cv2.imshow("video", im)
		cv2.waitKey(1)
