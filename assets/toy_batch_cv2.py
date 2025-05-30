# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
import cv2
import time

batch_size = 8
batch_indexes = np.arange(batch_size)
np.random.shuffle(batch_indexes)
cap = cv2.VideoCapture("/workspace/playbooks/video/assets/UCF101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c04.mp4")

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS) 
# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Video fps: {fps}, frames: {frame_count}")

tic = time.time()
frame_number = 0
success = True

while success and frame_number <= frame_count:

    for frame_idx in batch_indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number+frame_idx)
        success, image = cap.read()
    frame_number += fps

cap.release()

print("Iterated over the video in %f" % (time.time() - tic))