from openai import OpenAI
import os
import json
import glob
import re
import ast 
import requests

gt_path = "./Outputs/breakpoint/gpt-4o/gpt_answer.json"
eval_path = "./Outputs/dir_cot.json"
output_path = "./Outputs/eval_dir_cot_ernie.json"

client = OpenAI(
    base_url = "https://api.gptsapi.net/v1",
    api_key = "sk-6uJd1a3df931819cb0f3ecbb32b6e92115e798cc019CH1oJ"
)

with open(eval_path, 'r') as f:
    eval_files = json.load(f)

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=Bho8KOtpWZo7tOjgg3Udn2Tl&client_secret=7q4I99Svc5Yj8M4RwjWFWRqotZJvYygN"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

count = 0
mode = "ernie"
with open(gt_path, 'r') as f: 
    for line in f:
        count += 1
        # if count <= 98:
        #     continue
        outputs = {}
        video = json.loads(line)
        video_name = [key for key in video][0]
        outputs[video_name] = []
        for i, qa in enumerate(video[video_name]):
            output = {}
            question = qa["question"]
            gt_answer = qa["pred"]
            pred_answer = eval_files[video_name]["breakpoint"][i]["answer"]
            output["question"] = question
            output["gt"] = gt_answer
            output["pred"] = pred_answer
        
            prompt = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. " + \
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:" + \
                        "------" + \
                        "##INSTRUCTIONS: " + \
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n" + \
                        "- Consider synonyms or paraphrases as valid matches.\n" + \
                        "- Evaluate the correctness of the prediction compared to the answer." + \
                        "Please evaluate the following video-based question-answer pair:\n\n" + \
                        f"Question: {question}\n" + \
                        f"Correct Answer: {gt_answer}\n" + \
                        f"Predicted Answer: {pred_answer}\n\n" + \
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. " + \
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING." + \
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. " + \
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            if mode == "gpt":
                response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                        ]
                                    }
                                ],
                                max_tokens=200,
                            )
            
                result = response.choices[0].message.content
                
            else:
                url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k-0205?access_token=" + get_access_token()
                payload = json.dumps({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                })
            
                headers = {
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                result = response.json()
                result = result["result"]
                print(result)
                
            match = re.search(r"({.*})", result)
            if match:
                # match = re.sub(r"(\d)'", r"(\d)", match)
                result = ast.literal_eval(match.group(1))  
            else:
                continue
            
            print(result["pred"], result["score"])
            
            output["match"] = result["pred"]
            output["score"] = result["score"]
            
            outputs[video_name].append(output)
            # import sys
            # sys.exit(0)
        
        with open(output_path, 'a') as output_f:
            output_f.write(json.dumps(outputs))
            output_f.write("\n")            
            
            

num_correct = 0
num = 0
score = 0
with open(output_path, 'r') as f:
    for line in f:
        video = json.loads(line)
        video_name = [key for key in video][0]
        samples = video[video_name]
        for sample in samples:
            score += sample["score"]
            if sample["match"] == "yes":
                num_correct += 1
            num += 1

print(f"Acc:{num_correct/num*100:.2f}")
print(f"Score:{score/num:.2f}")
