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

"""
With a NGC pytorch container
additional requirements are 
pip install pycuda
pip install pynvvideocodec

to run it:
python3 pynvc_dataloader.py

Any video dataset would work to run it, you don't need UCF101
What you need is to preprocess such that frame size is the same
"""


import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import PyNvVideoCodec as nvc
import pycuda.driver as cuda
import torch 
import numpy as np

batch_size = 4
sequence_length = 8
stride = 1
initial_prefetch_size = 16
video_directory = "/workspace/playbooks/video/assets/output"

# Initialize CUDA
cuda.init()
device = cuda.Device(0)  # Use GPU 0
cuda_ctx = device.retain_primary_context()
cuda_ctx.push()
cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
cuda_stream_app = cuda.Stream()  

class PyNvCVideoDataset(Dataset):
    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []
        self.enable_async_allocations = False
        
        # Load video paths and labels from directory structure
        for label_idx, action in enumerate(sorted(os.listdir(root_dir))):
            action_dir = os.path.join(root_dir, action)
            if os.path.isdir(action_dir):
                for video_file in os.listdir(action_dir):
                    if video_file.endswith('.mp4'):  # Assuming videos are in .avi format
                        self.video_paths.append(os.path.join(action_dir, video_file))
                        self.labels.append(label_idx)

        demuxer = nvc.CreateDemuxer(self.video_paths[0])
        self.decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=demuxer.GetNvCodecId(),
            cudacontext=cuda_ctx.handle,
            cudastream=cuda_stream_nv_dec.handle,
            usedevicememory=True,
            enableasyncallocations=self.enable_async_allocations,
        )
                        
    def __len__(self):
        return len(self.video_paths)

    def load_batch(self, path_to_video, batch_indeces):
        # Initialize CUDA
        # cuda.init()
        # device = cuda.Device(0)  # Use GPU 0
        # cuda_ctx = device.retain_primary_context()
        # cuda_ctx.push()
        # cuda_stream_nv_dec = cuda.Stream()  # create cuda streams for allocations
        # cuda_stream_app = cuda.Stream()
    
        # start = cuda.Event()
        # end = cuda.Event()

        # start.record(stream=cuda_stream_nv_dec)
        demuxer = nvc.CreateDemuxer(path_to_video)
        # if self.decoder is None:
        #     self.decoder = nvc.CreateDecoder(
        #         gpuid=0,
        #         codec=demuxer.GetNvCodecId(),
        #         cudacontext=0, #cuda_ctx.handle,
        #         cudastream=0, #cuda_stream_nv_dec.handle,
        #         usedevicememory=True,
        #         enableasyncallocations=self.enable_async_allocations,
        #     )
    
        frame_count = 0
        frames = None
        for packet_index, packet in enumerate(demuxer):
            #print(packet_index)
            for frame in self.decoder.Decode(packet):
                frame_count += 1
                #print(frame_count)
                if frame_count in batch_indeces:
                    if frames is None:
                        #print(torch.from_dlpack(frame).size())
                        frames = torch.from_dlpack(frame).unsqueeze(0)
                        #frames = resize(torch.from_dlpack(frame), [360,320]).unsqueeze(0)
                        #print(frames.size())
                    else:
                        frames = torch.cat((frames, torch.from_dlpack(frame).unsqueeze(0)), dim=0)
        # decoder.WaitOnCUStream(cuda_stream_app.handle) # not sure if we need this
        # if frames.size(0) == len(batch_indeces):
        # end.record()
        # end.synchronize()
        # print("time %f" %(start.time_till(end)))
        # cuda_ctx.pop()
        return frames
        
    
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        #print(video_path)
        
        # Open a demuxer
        demuxer = nvc.CreateDemuxer(video_path) 
        # check video lenght
        for frame_count, packet in enumerate(demuxer):
            continue
        
        # Get frames
        #print(frame_count)
        frames_idx = np.floor(np.linspace(1, frame_count, sequence_length))
        #print(frames_idx)
        frames = self.load_batch(video_path, frames_idx) # Returns all frames
        # Apply any transformations (e.g., resizing, normalization)
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label

  

ucf101_dataset = PyNvCVideoDataset(root_dir=video_directory, transform=None)
dataloader = DataLoader(ucf101_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print(len(ucf101_dataset))
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i, data in enumerate(dataloader):
    video = data[0]
    label = data[1]
    #print(video.size())
    #print(label.size())
    if i == 9:
        break

start.record()
for i, data in enumerate(dataloader):
    if i == 10:
        break
    video = data[0]
    label = data[1]
    #print(video.size())
    #print(label.size())

end.record()
torch.cuda.synchronize()

decord_gpu_time = start.elapsed_time(end)/1000 # is is in ms
decord_gpu_time_batch = decord_gpu_time / i
print('It took %f seconds for %d batches, or %f second per batch' %(decord_gpu_time, i, decord_gpu_time_batch) )
cuda_ctx.pop()
