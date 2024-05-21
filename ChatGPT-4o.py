from openai import OpenAI
from PIL import Image
import os
import json
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

qa_folder = "./MovieChat-1K-test/annotations/"
file_list = os.listdir(qa_folder)
json_files = [filename for filename in file_list if filename.endswith('.json')]
output_file = "./Outputs/gpt_answer.json"

# clip_model = CLIPModel.from_pretrained()
# clip_processor = CLIPProcessor.from_pretrained()

# openai.api_key = "sk-6uJd1a3df931819cb0f3ecbb32b6e92115e798cc019CH1oJ"
# openai.api_base = "https://api.gptsapi.net/v1"

client = OpenAI(
    base_url = "https://api.gptsapi.net/v1",
    api_key = "sk-6uJd1a3df931819cb0f3ecbb32b6e92115e798cc019CH1oJ"
)

global_value = []

# prompt = "Here is the question: " + "What object appears?" + " Answer the question according to the image in less than 20 words."
# base64_image = encode_image("/home/sunyw/MovieChat/src/output_frame/2.mp4_0.jpg")
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#             ]
#         }
#     ],
#     max_tokens=200,
# )
# print(response.json())
# import sys
# sys.exit(0)

for file in json_files:
    if file.endswith(".json"):
        file_path = os.path.join(qa_folder, file)
        with open(file_path, 'r') as json_file:
            movie_data = json.load(json_file)
            global_key = movie_data["info"]["video_path"]
            print(f"Video Name:{global_key}\n")
            for qa_key in movie_data["breakpoint"]:
                cur_frame = qa_key["time"]
                question = qa_key["question"]
                temp_frame_path = './src/output_frame/'+ global_key + f"_{cur_frame}.jpg"
                base64_image = encode_image(temp_frame_path)
                prompt = "Here is the question: " + question + " Answer the question according to the image in less than 20 words."
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ]
                        }
                    ],
                    max_tokens=200,
                )
                # answer = response.json()
                # print(answer)
                qa_key["pred"] = response.choices[0].message.content#answer["choices"][0]["message"]["content"]
                global_value.append(qa_key)
                print(f"Question:{question}\nAnswer:"+qa_key["pred"]+"\n")
            result_data = {}
            result_data[global_key] = global_value
            with open(output_file, 'a') as output_json_file:
                            output_json_file.write(json.dumps(result_data))
                            output_json_file.write("\n")
            