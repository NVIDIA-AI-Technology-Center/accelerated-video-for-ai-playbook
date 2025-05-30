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
import av
import time

# Open the video file
container = av.open("/workspace/playbooks/video/assets/UCF101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c04.mp4")

fps = container.streams.video[0].average_rate
frame_count = container.streams.video[0].frames
print("Video fps: %f, frames: %d" % (fps, frame_count))
# Get the video stream
stream = container.streams.video[0]

tic = time.time()
# Calculate the frame interval (e.g., every 10 frames)
frame_interval = 10

# Decode and process frames
for frame_index, frame in enumerate(container.decode(stream)):
    if frame_index % frame_interval == 0:
        # Process the frame (e.g., save or display)
        image = frame.to_image()
        # image.save(f"frame-{frame_index}.jpg")
        # print(f"Processed frame {frame_index}")

# Close the container
container.close()
print("Iterated over the video in %f" % (time.time() - tic))