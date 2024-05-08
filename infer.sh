HF_ENDPOINT=https://hf-mirror.com
python inference.py \
    --cfg-path eval_configs/MovieChat.yaml \
    --gpu-id 0 \
    --num-beams 1 \
    --temperature 1.0 \
    --text-query "what happens?" \
    --video-path MovieChat-1K-test/videos/AWD-2.mp4 \
    --fragment-video-path src/video_fragment/output.mp4 \
    --cur-min 7 \
    --cur-sec 18 \
    --middle-video 1