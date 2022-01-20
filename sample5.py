import torch


Z = torch.tensor([[1, 2,3,4,5,6,7,8,9,10], 
                 [1, 2,3,4,5,6,7,8,9,10]], dtype=torch.float32)

C_pred_expand = torch.repeat_interleave(Z, 2, dim=1)
print(C_pred_expand)






#         import depthai as dai
# import cv2

# # Create pipeline
# pipeline = dai.Pipeline()

# # Define source and output
# camRgb = pipeline.create(dai.node.ColorCamera)
# xoutVideo = pipeline.create(dai.node.XLinkOut)

# xoutVideo.setStreamName("video")

# # Properties
# camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setVideoSize(1280, 720)

# xoutVideo.input.setBlocking(False)
# xoutVideo.input.setQueueSize(100)

# # Linking
# camRgb.video.link(xoutVideo.input)


# class VideoReader(object):
#     def __init__(self, file_name):
#         self.file_name = file_name

#     def __iter__(self):
#         with dai.Device(pipeline) as device:
#             self.cap = device.getOutputQueue(name="video", maxSize=500, blocking=False)
#             return self

#     def __next__(self):
#         videoIn = self.cap.get()
#         img = videoIn.getCvFrame()
#         return img

# vr = VideoReader("okd")
# for img in vr:
#     print(img.shape)