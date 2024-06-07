import clip
import json
import os 
import glob
from PIL import Image
image_ans_path = "/home/sunyw/MovieChat/Outputs/breakpoint/gpt-4o/gpt_answer.json"
video_ans_path = "/home/sunyw/MovieChat/Outputs/breakpoint/llama2+short+new_cot+clipsample_inter1+currange+knn+icl"
output_path = "/home/sunyw/MovieChat/Outputs/merge.json"

def get_ans_dict(path):
    videos_dict = {}
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(video_ans_path, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    video_dict = json.loads(line)
                    for key, value in video_dict.items():
                        videos_dict[key] = value
    else:
        with open(path, 'r') as f:
            for line in f:
                video_dict = json.loads(line)
                for key, value in video_dict.items():
                    videos_dict[key] = value
    return videos_dict

def control_length(sentence):
    sentence_list = sentence.split()
    if len(sentence_list) < 50:
        return sentence
    else:
        sentence = ""
        for i in range(50):
            sentence = sentence + " " + sentence_list[i]
        return sentence

def get_sorted_files(directory, pattern="*9.MP4*"):
    """
    使用 glob 搜索包含特定模式的文件，并按文件名排序
    :param directory: 要搜索的目录
    :param pattern: 文件名中包含的模式
    :return: 排序后的文件列表
    """
    # 构建搜索路径
    search_path = os.path.join(directory, pattern)
    
    # 使用 glob 搜索文件
    files = glob.glob(search_path)
    
    # 按文件名排序
    files.sort(key=lambda x: os.path.getctime(x))
    
    return files

model, processor = clip.load("./ckpt/ViT-B-32.pt", device="cuda:0")

image_ans_dict = get_ans_dict(image_ans_path)
video_ans_dict = get_ans_dict(video_ans_path)


for key, value in image_ans_dict.items():
    print(key)
    video_qa = video_ans_dict[key]
    image_files = get_sorted_files("/home/sunyw/MovieChat/src/output_frame", key+"*")
    global_value = []
    count = 0
    for id, qa in enumerate(value):
        ans_image = control_length(qa["pred"])
        ans_video = control_length(video_qa[id]["pred"])
        ans = [ans_image] + [ans_video]
        text = clip.tokenize(ans)
        try:
            time = qa["time"]
            image_file = "/home/sunyw/MovieChat/src/output_frame/" + f"{key}_{time}.jpg" 
            raw_image = Image.open(image_file).convert("RGB")
        except:
            image_file = image_files[id]
            raw_image = Image.open(image_file).convert("RGB")
        image = processor(raw_image).unsqueeze(0)
        logits_per_image, logits_per_text = model(image.to("cuda:0"), text.to("cuda:0"))
        probs = logits_per_image.softmax(dim=-1)
        if probs[0,0] < probs[0,1]:
            qa["pred"] = ans_video
            count += 1
        global_value.append(qa)
    result_data = {}
    result_data[key] = global_value
    
    with open(output_path, 'a') as f:
        f.write(json.dumps(result_data))
        f.write("\n")

print(f"video win: {count}")
        

                        
    