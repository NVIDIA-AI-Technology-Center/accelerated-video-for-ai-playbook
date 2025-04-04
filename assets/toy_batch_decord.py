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
from decord import VideoReader
from decord import cpu, gpu
import time

ctx = cpu()

batch_size = 8
batch_indexes = np.arange(batch_size)
np.random.shuffle(batch_indexes)
vr = VideoReader("/workspace/playbooks/video/assets/UCF101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c04.mp4", ctx)

# Get the frames per second
fps = vr.get_avg_fps()
# Get the total numer of frames in the video.
frame_count = len(vr)
print("Video fps: %d, frames: %d" % (fps, frame_count))

tic = time.time()
frame_number = 0

while frame_number <= (frame_count - batch_size):

    frame = vr.get_batch(batch_indexes+frame_number)
    frame_number += int(fps)

print("Iterated over the video in %f" % (time.time() - tic))
