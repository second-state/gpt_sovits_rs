with open("polyphonic-fix.rep") as f:
    polyphonic_fix = f.read()

lines = polyphonic_fix.split("\n")

word_dict = {}

for line in lines:
    item = line.split(":")
    key = item[0]
    value = item[1]
    value = eval(value)
    if "暴" in key:
        print("skip", key)
        continue
    word_dict[key] = value

import json

json.dump(word_dict, open("zh_word_dict.json", "w"), ensure_ascii=False, indent=2)
