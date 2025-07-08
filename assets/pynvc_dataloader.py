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
additional requirements is 
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
import torch 
import numpy as np

batch_size = 4
sequence_length = 8
stride = 1
initial_prefetch_size = 16
video_directory = "/workspace/playbooks/video/UCF-101/UCF-101_proc"

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

        # https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html#simpledecoder-samples
        self.decoder = nvc.SimpleDecoder(
            enc_file_path=self.video_paths[0],        # Input filename 
            gpu_id=0,                                 # Index of GPU, useful for multi-GPU setups 
            use_device_memory=True,                   # Decoded frames reside in device memory
            max_width=320,                           # Maximum width of buffer for decoder reuse 
            max_height=240,                          # Maximum height of buffer for decoder reuse 
            need_scanned_stream_metadata=False,        # Retrieve stream-level metadata 
            output_color_type=nvc.OutputColorType.RGB # Decoded frames available as RGB or YUV
        )
                        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Create decoder with GPU output
        self.decoder.reconfigure_decoder(video_path)
        frame_count = len(self.decoder)
        
        # Get frames
        # marginally faster with .tolist()
        frames_idx = np.linspace(1, frame_count-1, sequence_length, dtype=int).tolist()
        
        # # Get multiple frames at regular intervals
        self.decoder.seek_to_index(0) # make sure to start from frame 0
        frames = self.decoder.get_batch_frames_by_index(frames_idx)

        # Convert all frames to tensors (all zero-copy)
        tensors = [torch.from_dlpack(frame) for frame in frames]
        # Stack into a batch tensor
        batch = torch.stack(tensors)

        # Apply any transformations (e.g., resizing, normalization)
        if self.transform:
            batch = self.transform(batch)
        
        return batch, label

  

ucf101_dataset = PyNvCVideoDataset(root_dir=video_directory, transform=None)
dataloader = DataLoader(ucf101_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(len(ucf101_dataset))
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("----- warmup -----")
for i, data in enumerate(dataloader):
    video = data[0]
    label = data[1]
    #print(video.size())
    #print(label.size())
    if i == 9:
        break
print("--- end warmup ---")

print("-- start timing --")
start.record()
for i, data in enumerate(dataloader):
    if i == 100:
        break
    video = data[0]
    label = data[1]
    #print(video.size())
    #print(label.size())

end.record()
torch.cuda.synchronize()
print("--- end timing ---")

gpu_time = start.elapsed_time(end)/1000 # is is in ms
gpu_time_batch = gpu_time / (i* batch_size * sequence_length) # in seconds per frame
print('It took %f seconds for %d batches of %d frames, or %f second per frame' %(gpu_time, i, sequence_length, gpu_time_batch) )

