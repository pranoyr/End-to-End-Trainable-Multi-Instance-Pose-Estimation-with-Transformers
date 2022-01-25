import torch


# Z = torch.tensor([[1, 2,3,4,5,6,7,8,9,10], 
#                  [1, 2,3,4,5,6,7,8,9,10]], dtype=torch.float32)

# C_pred_expand = torch.repeat_interleave(Z, 2, dim=1)
# print(C_pred_expand)



a = torch.tensor([[1,2,3],
                    [4,5,6]], dtype=torch.float32)


b = torch.tensor([[6 ,7,8],
                    [9,10,11]], dtype=torch.float32)


viz_loss  =  torch.cdist(a, b, p=2)

print(viz_loss.square())



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

Epoch: [0]  [   60/14150]  eta: 1:09:44  lr: 0.000100  class_error: 75.00  loss: 460.6976 (769.2592)  loss_bbox: 74.1521 (125.0458)  loss_bbox_0: 78.5719 (132.4391)  loss_bbox_1: 77.4377 (128.9973)  loss_bbox_2: 76.7487 (126.9186)  loss_bbox_3: 75.1801 (126.1018)  loss_bbox_4: 74.4349 (125.3347)  loss_ce: 0.4316 (0.6408)  loss_ce_0: 0.5969 (0.7900)  loss_ce_1: 0.5951 (0.8358)  loss_ce_2: 0.5293 (0.7656)  loss_ce_3: 0.4893 (0.7100)  loss_ce_4: 0.4409 (0.6798)  abs_loss_unscaled: 2.4058 (5.0817)  abs_loss_0_unscaled: 2.7319 (5.4223)  abs_loss_1_unscaled: 2.6535 (5.2644)  abs_loss_2_unscaled: 2.5402 (5.1670)  abs_loss_3_unscaled: 2.4558 (5.1302)  abs_loss_4_unscaled: 2.4194 (5.0950)  cardinality_error_unscaled: 2.7500 (38.2295)  cardinality_error_0_unscaled: 30.0000 (63.4959)  cardinality_error_1_unscaled: 28.7500 (64.7049)  cardinality_error_2_unscaled: 14.5000 (54.2295)  cardinality_error_3_unscaled: 12.0000 (48.9057)  cardinality_error_4_unscaled: 4.5000 (44.0820)  center_loss_unscaled: 0.0396 (0.0563)  center_loss_0_unscaled: 0.0452 (0.0465)  center_loss_1_unscaled: 0.0445 (0.0491)  center_loss_2_unscaled: 0.0420 (0.0505)  center_loss_3_unscaled: 0.0430 (0.0526)  center_loss_4_unscaled: 0.0388 (0.0556)  class_error_unscaled: 90.0000 (45.7271)  loss_bbox_unscaled: 14.8304 (25.0092)  loss_bbox_0_unscaled: 15.7144 (26.4878)  loss_bbox_1_unscaled: 15.4875 (25.7995)  loss_bbox_2_unscaled: 15.3497 (25.3837)  loss_bbox_3_unscaled: 15.0360 (25.2204)  loss_bbox_4_unscaled: 14.8870 (25.0669)  loss_ce_unscaled: 0.4316 (0.6408)  loss_ce_0_unscaled: 0.5969 (0.7900)  loss_ce_1_unscaled: 0.5951 (0.8358)  loss_ce_2_unscaled: 0.5293 (0.7656)  loss_ce_3_unscaled: 0.4893 (0.7100)  loss_ce_4_unscaled: 0.4409 (0.6798)  offset_loss_unscaled: 7.2663 (8.8841)  offset_loss_0_unscaled: 7.5566 (9.1281)  offset_loss_1_unscaled: 7.4734 (9.0125)  offset_loss_2_unscaled: 7.3991 (8.9588)  offset_loss_3_unscaled: 7.2055 (8.9234)  offset_loss_4_unscaled: 7.2714 (8.8940)  vis_loss_unscaled: 1.0358 (1.0613)  vis_loss_0_unscaled: 1.0283 (1.0561)  vis_loss_1_unscaled: 1.0221 (1.0
