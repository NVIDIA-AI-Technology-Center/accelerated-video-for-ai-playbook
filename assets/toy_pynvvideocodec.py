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

import PyNvVideoCodec as nvc
import pycuda.driver as cuda
import torch 

enable_async_allocations = False
# Initialize CUDA
cuda.init()
device = cuda.Device(0)  # Use GPU 0
cuda_ctx = device.retain_primary_context()
cuda_ctx.push()
cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
cuda_stream_app = cuda.Stream()

# File path to the video
video_path = "/workspace/playbooks/video/assets/UCF101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c04.mp4"

# Create a demuxer for extracting packets from the video file
demuxer = nvc.CreateDemuxer(video_path)

# Create a decoder for decoding video packets into raw frames
decoder = nvc.CreateDecoder(
    gpuid=0, 
    codec=demuxer.GetNvCodecId(), #nvc.cudaVideoCodec.H264,  
    cudacontext=cuda_ctx.handle,
    cudastream=cuda_stream_nv_dec.handle,  # Default CUDA stream
    usedevicememory=True,  # Use GPU memory for decoded frames
    enableasyncallocations=enable_async_allocations
)

# Get video properties (FPS and frame count)
fps = demuxer.FrameRate

# frame_count, is there no way to find the frame count? 
# print(f"Video fps: {fps}, frames: {frame_count}")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
frame_number = 0

start.record()
# Iterate over packets and decode them into frames
for packet in demuxer:
    for decoded_frame in decoder.Decode(packet):
        frame_number += 1
        # Process the decoded frame (decoded_frame is in GPU memory)
        # src_tensor = torch.from_dlpack(decoded_frame)
end.record()
torch.cuda.synchronize()

time = start.elapsed_time(end)/1000 # is is in ms
print(f"Iterated over the video in {time} seconds, video lenght in frames:{frame_number}")

# Release resources
#decoder.CLose()
#demuxer.Close()
cuda_ctx.pop()
#cuda_ctx.retain_primary_context().detach()