python cpsl_preprocess.py   --input-video input/monkey.mp4   --output-root data/output_vod/monkey   --depth-checkpoint ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth   --depth-repo ../Depth-Anything-V2

python cpsl_playback.py --input data/output_vod/monkey --random-seed 12 --output data/output_vod/monkey/playback.mp4