import json
with open("/home/sunyw/MovieChat/Outputs/breakpoint/internvl+cot/update_prompt.json", "r") as f:
    old_json_file = json.load(f)

for key, value in old_json_file.items():
    item = {}
    key = key+".mp4"
    item[key] = []
    for answer in value["breakpoint"]:
        item[key].append({"question":answer["question"], "time":answer["time"], "pred":answer["answer"]})
    with open("/home/sunyw/MovieChat/Outputs/breakpoint/internvl+cot/breakpoint.json", "a") as f:
        f.write(json.dumps(item))
        f.write("\n")