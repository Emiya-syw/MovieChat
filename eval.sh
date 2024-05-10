HF_ENDPOINT=https://hf-mirror.com

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    # if [ $IDX -eq 0 ]; then
    #     continue
    # fi
    # if [ $IDX -eq 1 ]; then
    #     continue
    # fi
    # if [ $IDX -eq 2 ]; then
    #     continue
    # fi  
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval_code/result_prepare/run_inference_qa_moviechat.py \
        --cfg-path ./eval_configs/MovieChat.yaml \
        --gpu-id 0 \
        --num-beams 1 \
        --temperature 1.0 \
        --video-folder ./MovieChat-1K-test/videos \
        --qa-folder ./MovieChat-1K-test/annotations \
        --output-dir ./Outputs \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --fragment-video-path ./Outputs/interoutput_${IDX}.mp4 \
        --middle-video 0&
done

wait

echo "All tasks are completed."