import os
import json

bkpt_path = './Outputs/breakpoint/llama2+range+cot'
glb_path = './Outputs/global/llama2+cot'
output_path = "./Outputs/sub0513.json"
submit_dict = {}

# breakpoint
for filename in os.listdir(bkpt_path):
    if filename.endswith('.json'):
        file_path = os.path.join(bkpt_path, filename)
        # print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                video_dict = json.loads(line)
                video_key = [key for key in video_dict][0]  # 视频文件名
                submit_dict[video_key] = {}                 # 若从submit_dict非空，就不需要初始化
                bkpt_list = []
                for qa_dict in video_dict[video_key]:
                    # print(qa_dict)
                    new_qa_dict = {}
                    new_qa_dict['question'] = qa_dict['question']
                    new_qa_dict['answer'] = qa_dict['pred']
                    new_qa_dict['time'] = qa_dict['time']
                    bkpt_list.append(new_qa_dict)
                # submit_dict[video_key]['caption'] = video_key  # captain的来源比赛未说明清楚，这里使用文件名代替
                submit_dict[video_key]['caption'] = '...'  # captain的来源比赛未说明清楚，这里使用'...'代替
                submit_dict[video_key]['breakpoint'] = bkpt_list


# global
for filename in os.listdir(glb_path):
    if filename.endswith('.json'):
        file_path = os.path.join(glb_path, filename)
        # print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                video_dict = json.loads(line)
                video_key = [key for key in video_dict][0]  # 视频文件名
                glb_list = []
                for qa_dict in video_dict[video_key]:
                    # print(qa_dict)
                    new_qa_dict = {}
                    new_qa_dict['question'] = qa_dict['question']
                    new_qa_dict['answer'] = qa_dict['pred']
                    glb_list.append(new_qa_dict)

                submit_dict[video_key]['global'] = glb_list

# write data
with open(output_path, "w") as f:
        json.dump(submit_dict, f)