python cpsl_preprocess.py   --input-video input/monkey.mp4   --output-root data/output_vod/monkey   --depth-checkpoint ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth   --depth-repo ../Depth-Anything-V2

python cpsl_playback.py --input data/output_vod/monkey --random-seed 12 --output data/output_vod/monkey/playback.mp4

python cpsl_playback.py \
  --input data/output_vod/malaysia \
  --output data/output_vod/malaysia/playback_orbit.mp4 \
  --trace traces/viewport_orbit.csv


python cpsl_preprocess.py   --input-video input/monkey.mp4   --output-root data/output_vod/monkey   --depth-checkpoint ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth   --depth-repo ../Depth-Anything-V2   --target-layers 6



python cpsl_playback.py \
  --input data/output_vod/Accusefive \
  --output data/output_vod/Accusefive/playback_orbit.mp4 \
  --trace traces/viewport_orbit.csv