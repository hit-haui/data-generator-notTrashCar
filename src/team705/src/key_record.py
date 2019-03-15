import keyboard
import time
import json

# keyboard.name
# keyboard.time
# keyboard.event_type

output_list = keyboard.record()
cur_time = start_time = output_list[0].time
idx = 1
record = {} 
key_left = key_right = key_down = key_up = False

while idx < len(output_list) - 1:
    if cur_time >= output_list[idx].time:
        cur_time = output_list[idx].time
        # key_down
        if output_list[idx].name == 'up' and output_list[idx].event_type == 'down':
            key_up = True
        elif output_list[idx].name == 'down' and output_list[idx].event_type == 'down':
            key_down = True
        elif output_list[idx].name == 'right' and output_list[idx].event_type == 'down':
            key_right = True
        elif output_list[idx].name == 'left' and output_list[idx].event_type == 'down':
            key_left = True
        # key_up
        if output_list[idx].name == 'up' and output_list[idx].event_type == 'up':
            key_up = False
        elif output_list[idx].name == 'down' and output_list[idx].event_type == 'up':
            key_down = False
        elif output_list[idx].name == 'right' and output_list[idx].event_type == 'up':
            key_right = False
        elif output_list[idx].name == 'left' and output_list[idx].event_type == 'up':
            key_left = False
        idx += 1
    else:
        cur_time += 0.001 
    if idx == len(output_list) - 1:
        break
    record[cur_time - start_time] = []
    record[cur_time - start_time].append({
        # 'time' : each.time - start_time,
        'up' : key_up,
        'down' : key_down,
        'right' : key_right,
        'left' : key_left,
    })
with open('key_data.json', 'w', encoding = 'utf-8') as outfile:  
        json.dump(record, outfile, ensure_ascii= False,sort_keys= False, indent=4)
        outfile.write("\n")