#!/usr/bin/env sh

# Create an animated movie from a series or folder of images using FFmpeg.

ffmpeg -start_number 0 -i img_%d.png -framerate 25 -pix_fmt yuv420p growth.mp4
