import json

with open("/home/sunyw/MovieChat/Outputs/submission0607.json","r") as f:
    file = json.load(f)

for key, value in file.items():
    if len(value["global"]) == 3:
        gb_num = "OK"
    else:
        gb_num = len(value["global"])
    if len(value["breakpoint"]) >= 10:
        bk_num = "OK"
    else:
        bk_num = len(value["breakpoint"])
    print(f"GB:{gb_num}, BK:{bk_num}")