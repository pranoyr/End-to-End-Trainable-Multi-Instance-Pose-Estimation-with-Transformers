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
Epoch: [0]  [    0/14150]  eta: 3:11:22  lr: 0.000100  class_error: 0.00  loss: 1627.8856 (1627.8856)  loss_bbox: 270.9528 (270.9528)  loss_bbox_0: 271.0305 (271.0305)  loss_bbox_1: 271.0029 (271.0029)  loss_bbox_2: 268.4996 (268.4996)  loss_bbox_3: 270.5366 (270.5366)  loss_bbox_4: 270.4482 (270.4482)  loss_ce: 0.9360 (0.9360)  loss_ce_0: 0.8879 (0.8879)  loss_ce_1: 0.9035 (0.9035)  loss_ce_2: 0.9044 (0.9044)  loss_ce_3: 0.8747 (0.8747)  loss_ce_4: 0.9088 (0.9088)  cardinality_error_unscaled: 95.2500 (95.2500)  cardinality_error_0_unscaled: 94.5000 (94.5000)  cardinality_error_1_unscaled: 95.0000 (95.0000)  cardinality_error_2_unscaled: 95.2500 (95.2500)  cardinality_error_3_unscaled: 95.0000 (95.0000)  cardinality_error_4_unscaled: 95.2500 (95.2500)  class_error_unscaled: 0.0000 (0.0000)  loss_abs_unscaled: 12.0744 (12.0744)  loss_abs_0_unscaled: 12.0788 (12.0788)  loss_abs_1_unscaled: 12.0833 (12.0833)  loss_abs_2_unscaled: 11.9595 (11.9595)  loss_abs_3_unscaled: 12.0558 (12.0558)  loss_abs_4_unscaled: 12.0501 (12.0501)  loss_bbox_unscaled: 54.1906 (54.1906)  loss_bbox_0_unscaled: 54.2061 (54.2061)  loss_bbox_1_unscaled: 54.2006 (54.2006)  loss_bbox_2_unscaled: 53.6999 (53.6999)  loss_bbox_3_unscaled: 54.1073 (54.1073)  loss_bbox_4_unscaled: 54.0896 (54.0896)  loss_ce_unscaled: 0.9360 (0.9360)  loss_ce_0_unscaled: 0.8879 (0.8879)  loss_ce_1_unscaled: 0.9035 (0.9035)  loss_ce_2_unscaled: 0.9044 (0.9044)  loss_ce_3_unscaled: 0.8747 (0.8747)  loss_ce_4_unscaled: 0.9088 (0.9088)  loss_center_unscaled: 0.0105 (0.0105)  loss_center_0_unscaled: 0.0106 (0.0106)  loss_center_1_unscaled: 0.0106 (0.0106)  loss_center_2_unscaled: 0.0109 (0.0109)  loss_center_3_unscaled: 0.0107 (0.0107)  loss_center_4_unscaled: 0.0105 (0.0105)  loss_offset_unscaled: 11.3482 (11.3482)  loss_offset_0_unscaled: 11.3432 (11.3432)  loss_offset_1_unscaled: 11.2969 (11.2969)  loss_offset_2_unscaled: 11.2874 (11.2874)  loss_offset_3_unscaled: 11.3258 (11.3258)  loss_offset_4_unscaled: 11.3393 (11.3393)  loss_vis_unscaled: 1.0676 (1.0676)  loss_vis_0_unscaled: 1.0699 (1.0699)  loss_vis_1_unscaled: 1.0686 (1.0686)  loss_vis_2_unscaled: 1.0637 (1.0637)  loss_vis_3_unscaled: 1.0785 (1.0785)  loss_vis_4_unscaled: 1.0719 (1.0719)  time: 0.8115  data: 0.4597  max mem: 2823
Epoch: [0]  [   10/14150]  eta: 1:21:09  lr: 0.000100  class_error: 0.00  loss: 1392.0498 (1321.0700)  loss_bbox: 229.9348 (217.2694)  loss_bbox_0: 233.8083 (222.9171)  loss_bbox_1: 232.2996 (220.7563)  loss_bbox_2: 230.2728 (218.3891)  loss_bbox_3: 229.8991 (218.1932)  loss_bbox_4: 229.7419 (217.4569)  loss_ce: 1.0019 (0.9896)  loss_ce_0: 1.0196 (1.0217)  loss_ce_1: 1.0439 (1.0695)  loss_ce_2: 0.9971 (1.0249)  loss_ce_3: 0.9788 (0.9761)  loss_ce_4: 1.0110 (1.0061)  cardinality_error_unscaled: 95.2500 (95.2955)  cardinality_error_0_unscaled: 94.5000 (93.7955)  cardinality_error_1_unscaled: 97.2500 (97.0000)  cardinality_error_2_unscaled: 95.7500 (95.8636)  cardinality_error_3_unscaled: 95.0000 (94.6136)  cardinality_error_4_unscaled: 96.0000 (95.8182)  class_error_unscaled: 0.0000 (3.7879)  loss_abs_unscaled: 10.1488 (9.4424)  loss_abs_0_unscaled: 10.3304 (9.7143)  loss_abs_1_unscaled: 10.2708 (9.6144)  loss_abs_2_unscaled: 10.1678 (9.4993)  loss_abs_3_unscaled: 10.1453 (9.4874)  loss_abs_4_unscaled: 10.1395 (9.4512)  loss_bbox_unscaled: 45.9869 (43.4539)  loss_bbox_0_unscaled: 46.7617 (44.5834)  loss_bbox_1_unscaled: 46.4599 (44.1513)  loss_bbox_2_unscaled: 46.0546 (43.6778)  loss_bbox_3_unscaled: 45.9798 (43.6386)  loss_bbox_4_unscaled: 45.9484 (43.4914)  loss_ce_unscaled: 1.0019 (0.9896)  loss_ce_0_unscaled: 1.0196 (1.0217)  loss_ce_1_unscaled: 1.0439 (1.0695)  loss_ce_2_unscaled: 0.9971 (1.0249)  loss_ce_3_unscaled: 0.9788 (0.9761)  loss_ce_4_unscaled: 1.0110 (1.0061)  loss_center_unscaled: 0.0125 (0.0238)  loss_center_0_unscaled: 0.0129 (0.0202)  loss_center_1_unscaled: 0.0124 (0.0216)  loss_center_2_unscaled: 0.0117 (0.0222)  loss_center_3_unscaled: 0.0124 (0.0236)  loss_center_4_unscaled: 0.0132 (0.0234)  loss_offset_unscaled: 10.7021 (10.9047)  loss_offset_0_unscaled: 10.8161 (10.9930)  loss_offset_1_unscaled: 10.7527 (10.9272)  loss_offset_2_unscaled: 10.7300 (10.9001)  loss_offset_3_unscaled: 10.7270 (10.9134)  loss_offset_4_unscaled: 10.7199 (10.9087)  loss_vis_unscaled: 1.0952 (1.1009)  loss_vis_0_unscaled: 1.0965 (1.0980)  loss_vis_1_unscaled: 1.1027 (1.0967)  loss_vis_2_unscaled: 1.1044 (1.0981)  loss_vis_3_unscaled: 1.1017 (1.1023)  loss_vis_4_unscaled: 1.1005 (1.1019)  time: 0.3444  data: 0.0454  max mem: 4429
