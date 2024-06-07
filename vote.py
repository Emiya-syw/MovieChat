import json
import random
file_1 = "/home/sunyw/MovieChat/Outputs/sub0601.json"
file_2 = "/home/sunyw/MovieChat/Outputs/sub0602.json"
file_3 = "/home/sunyw/MovieChat/Outputs/global/timechat/sorted_answer.json"
output = "/home/sunyw/MovieChat/Outputs/global/vote.json"

with open(file_1, 'r') as f:
    file_1_json = json.load(f)
    
with open(file_2, 'r') as f:
    file_2_json = json.load(f)
    
with open(file_3, 'r') as f:
    file_3_json = json.load(f)
    
def decide(yes_list, no_list, question):
    if len(yes_list) > len(no_list):
        return {"question":question, "pred":random.choice(yes_list)}
    else:
        print(len(no_list))
        return {"question":question, "pred":random.choice(no_list)}
    
for key, value in file_1_json.items():
    value_1 = value["global"]
    value_2 = file_2_json[key]["global"]
    value_3 = file_3_json[key]["global"]
    
    qa_list = []
    for i in range(len(value_1)):
        question_1 = value_1[i]["question"]
        answer_1 = value_1[i]["answer"]
        question_2 = value_2[i]["question"]
        answer_2 = value_2[i]["answer"]
        question_3 = value_3[i]["question"]
        answer_3 = value_3[i]["answer"]
        
        yes_list = []
        no_list = []
        if question_1 == "When does the video take place? Nowadays or ancient times?":
            def vote_era(answer, yes_list, no_list):
                if "takes place in ancient times" in answer.lower():
                    print(answer)
                    no_list.append(answer)
                else:
                    yes_list.append(answer)
            vote_era(answer_1, yes_list, no_list)
            vote_era(answer_2, yes_list, no_list)
            vote_era(answer_3, yes_list, no_list)
            qa = decide(yes_list, no_list, question_1)
            
        elif question_1 == "what time does the video take place?(at day time or night time?)":
            def vote_time(answer, yes_list, no_list):
                if "night" in answer.lower():
                    no_list.append(answer)
                else:
                    yes_list.append(answer)
            vote_time(answer_1, yes_list, no_list)
            vote_time(answer_2, yes_list, no_list)
            vote_time(answer_3, yes_list, no_list)
            qa = decide(yes_list, no_list, question_1)
            
        elif "Does" in question_1 or "Do" in question_1 or "Is" in question_1 or "Are" in question_1:
            def vote_yes_no(answer, yes_list, no_list):
                if "no" in answer.lower():
                    no_list.append(answer)
                else:
                    yes_list.append(answer)
            vote_yes_no(answer_1, yes_list, no_list)
            vote_yes_no(answer_2, yes_list, no_list)
            vote_yes_no(answer_3, yes_list, no_list)
            qa = decide(yes_list, no_list, question_1)
        else:
            qa = {"question":question_1, "pred": answer_3}
        
        qa_list.append(qa)
    
    result_data = {}
    result_data[key] = qa_list
    with open(output, 'a') as output_json_file:
        output_json_file.write(json.dumps(result_data))
        output_json_file.write("\n")
    
        