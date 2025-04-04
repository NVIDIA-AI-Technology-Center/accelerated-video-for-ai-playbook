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
import numpy as np

enable_async_allocations = False

batch_size = 8
batch_indexes = np.arange(batch_size, dtype=np.int32)
#np.random.shuffle(batch_indexes)



# File path to the video
video_path = "/workspace/playbooks/video/assets/UCF101_processed/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c04.mp4"

def load_batch(path_to_video, batch_indeces):
    # Initialize CUDA
    cuda.init()
    device = cuda.Device(0)  # Use GPU 0
    cuda_ctx = device.retain_primary_context()
    cuda_ctx.push()
    cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
    cuda_stream_app = cuda.Stream()

    demuxer = nvc.CreateDemuxer(path_to_video)
    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=demuxer.GetNvCodecId(),
        cudacontext=cuda_ctx.handle,
        cudastream=cuda_stream_nv_dec.handle,
        usedevicememory=True,
        enableasyncallocations=enable_async_allocations,
    )

    frame_count = 0
    frames = []
    for  packet in demuxer:
        for frame in decoder.Decode(packet):
            frame_count += 1
            decoder.WaitOnCUStream(cuda_stream_app.handle)
            if frame_count in batch_indeces:
                frames.append(frame)
                if len(frames) == len(batch_indeces):
                    return frames
    
    cuda_ctx.pop()
    
# # Initialize CUDA
# cuda.init()
# device = cuda.Device(0)  # Use GPU 0
# cuda_ctx = device.retain_primary_context()
# cuda_ctx.push()
# cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
# cuda_stream_app = cuda.Stream()

# Create a demuxer for extracting packets from the video file
demuxer = nvc.CreateDemuxer(video_path)

# # Create a decoder for decoding video packets into raw frames
# decoder = nvc.CreateDecoder(
#     gpuid=0, 
#     codec=demuxer.GetNvCodecId(), #nvc.cudaVideoCodec.H264,  
#     cudacontext=cuda_ctx.handle,
#     cudastream=cuda_stream_nv_dec.handle,  # Default CUDA stream
#     usedevicememory=True,  # Use GPU memory for decoded frames
#     enableasyncallocations=enable_async_allocations
# )

# Get video properties (FPS and frame count)
fps = demuxer.FrameRate()

# frame_count, is there no way to find the frame count? 
# print(f"Video fps: {fps}, frames: {frame_count}")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
frame_number = 0
frame_count = 0

start.record()
# Iterate over packets and decode them into frames
for frame_count, packet in enumerate(demuxer):
    continue
    #for _ in decoder.Decode(packet): # not needed
    #frame_counter += 1
    # Process the decoded frame (decoded_frame is in GPU memory)
    # src_tensor = torch.from_dlpack(decoded_frame)
print("number of frames {frame_counter}")

# cuda_ctx.pop()

start.record()
while frame_number <= frame_count:
    images = load_batch(video_path, batch_indexes + frame_number)
        # Process the frame (e.g., save or display)
        # image.save(f"frame-{frame_index}.jpg")
        # print(f"Processed frame {frame_index}")
    frame_number += fps
end.record()
torch.cuda.synchronize()

time = start.elapsed_time(end)/1000 # is is in ms
print(f"Iterated over the video in {time} seconds, video lenght in frames:{frame_count}")

# Release resources
#decoder.CLose()
#demuxer.Close()
#cuda_ctx.pop()
#cuda_ctx.retain_primary_context().detach()

"""
def decode(enc_file_path, frame_idx):
    nv_dmx = nvc.CreateDemuxer(filename=enc_file_path)
    nv_dec = nvc.CreateDecoder(gpuid=0,
                               codec=nv_dmx.GetNvCodecId(),
                               cudacontext=0,
                               cudastream=0,
                               usedevicememory=1)

    decoded_frame_size = 0
    frames = []
    idx = 0
    print("FPS = ", nv_dmx.FrameRate())
    for packet in nv_dmx:
        for decoded_frame in nv_dec.Decode(packet):
            if idx in frame_idx:
                decoded_frame_size = nv_dec.GetFrameSize()
                src_tensor = torch.from_dlpack(decoded_frame)
                frames.append(src_tensor)
            idx = idx + 1
    return frames
"""

"""
import random
import torch
import cvcuda
import PyNvVideoCodec as nvc
import pycuda.driver as cuda
import numpy as np

# Initialize CUDA
cuda.init()
device = cuda.Device(0)
context = device.retain_primary_context()
context.push()

# File path to the video
video_path = "/path/to/your/video.mp4"

# Parameters for batch processing
batch_size = 8  # Number of frames per batch
target_frame_height, target_frame_width = 224, 224  # Resize dimensions

# Create a demuxer and decoder
demuxer = nvc.CreateDemuxer(video_path)
decoder = nvc.CreateDecoder(
    gpuid=0,
    codec=nvc.cudaVideoCodec.H264,  # Adjust codec if necessary
    cudacontext=context.handle,
    cudastream=0,
    usedevicememory=True
)

# Get total number of frames in the video
total_frames = int(demuxer.GetProperty(nvc.DemuxerProperty.FrameCount))

# Generate a list of random frame indices to sample
random_frame_indices = sorted(random.sample(range(total_frames), total_frames))  # Sorted for sequential decoding

# Function to process a batch of frames and convert to PyTorch tensors
def process_batch(frames):
    # Convert frames to CV-CUDA tensors for GPU processing
    tensor_batch = [cvcuda.as_tensor(frame) for frame in frames]
    
    # Example operation: Resize frames using CV-CUDA
    resized_batch = cvcuda.resize(tensor_batch, (target_frame_width, target_frame_height))
    
    # Convert resized frames to PyTorch tensors (BATCH x CHANNELS x HEIGHT x WIDTH)
    torch_tensors = [torch.from_numpy(frame.cpu().numpy()) for frame in resized_batch]
    
    # Stack tensors into a batch tensor (BATCH x CHANNELS x HEIGHT x WIDTH)
    return torch.stack(torch_tensors)

# Load and process random batches of frames
batch_frames = []
processed_batches = []
frame_count = 0

current_index = 0  # Track the current frame index being processed

for packet in demuxer:
    for decoded_frame in decoder.Decode(packet):
        if current_index in random_frame_indices:
            # Convert GPU frame to CPU-accessible format (e.g., NumPy array)
            frame = np.array(decoded_frame.Plane(0))  # Assuming single-plane YUV or RGB format
            
            batch_frames.append(frame)
            frame_count += 1
            
            # If batch is full, process it
            if len(batch_frames) == batch_size:
                processed_batch = process_batch(batch_frames)
                processed_batches.append(processed_batch)  # Store processed batch
                
                print(f"Processed batch of {len(processed_batch)} frames")
                batch_frames.clear()  # Clear the batch for the next set of frames
        
        current_index += 1
        
        # Stop decoding if all selected frames are processed
        if current_index > max(random_frame_indices):
            break

# Process any remaining frames in the last batch
if batch_frames:
    processed_batch = process_batch(batch_frames)
    processed_batches.append(processed_batch)
    print(f"Processed final batch of {len(processed_batch)} frames")

print(f"Total frames processed: {frame_count}")

# Release resources
decoder.Close()
demuxer.Close()
context.pop()
context.retain_primary_context().detach()
"""