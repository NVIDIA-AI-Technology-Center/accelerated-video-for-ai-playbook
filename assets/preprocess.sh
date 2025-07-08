# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
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
#!/bin/bash

# navigate to your dataset folder
cd /path/where/the/dataset/rar/is/located

# you might need sudo apt install unrar
unrar x -opUCF-101 UCF101.rar

# create the output folder
mkdir -p UCF-101/UCF-101_proc
# create subfolders in the output folder
find UCF-101 -mindepth 2 -maxdepth 2 -type d ! -path 'UCF-101/UCF-101_proc*' -exec bash -c 'input="{}"; output="UCF-101/UCF-101_proc/${input:16}"; mkdir -p "$output"' \;

# preprocess the videos
# added multiprocessing to speed up
find UCF-101 -mindepth 2 -maxdepth 3 -name "*.avi" ! -path 'UCF-101/UCF-101_proc*' \
  -print0 | xargs -0 -P 16 -I {} bash -c '
    input={};
    output="UCF-101/UCF-101_proc/${input:16:-3}mp4";
    ffmpeg -hide_banner -loglevel error -i "$input" -vf scale=320:240 -c:v libx264 -preset slow -crf 22 -an -r 30 "$output"
'
