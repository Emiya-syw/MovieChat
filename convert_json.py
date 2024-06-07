import json
with open("/home/sunyw/MovieChat/Outputs/global/timechat/sorted_answer.json", "r") as f:
    old_json_file = json.load(f)

for key, value in old_json_file.items():
    item = {}
    # key = key+".mp4"
    item[key] = []
    for answer in value["global"]:
        item[key].append({"question":answer["question"], "pred":answer["answer"]})
    with open("/home/sunyw/MovieChat/Outputs/global/timechat/merged_answer.json", "a") as f:
        f.write(json.dumps(item))
        f.write("\n")