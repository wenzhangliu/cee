import numpy as np
# import torch
# torch.set_num_threads(8)
import time
import cv2

buffer_path = None

origin_obs = np.load(buffer_path)
for index in range(len(origin_obs)):
    pic = origin_obs[index,:,:,:]
    # pic = pic.transpose(1,2,0)
    # pic = pic.transpose(2, 1, 0)
    pic = pic[:, :, ::-1]
    pic =cv2.resize(pic,(880,480))
    print(index)
    cv2.imshow("1",pic)
    # print(pic)
    # cv2.waitKey()
    cv2.waitKey(1000)

