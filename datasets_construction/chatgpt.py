from openai import OpenAI
import os
import json
import ast
import math

def delete_elements_by_indices(lst, indices):
    indices = sorted(indices, reverse=True)

    for index in indices:
        if 0 <= index < len(lst):
            del lst[index]

def location(arc):
    degree = round(math.degrees(arc))
    if degree < 0:
        loc = '{}'.format(abs(degree)) + ' degrees northeast' #右边
        return loc
    elif degree == 0:
        loc = 'north'
        return loc
    else:
        loc = '{}'.format(abs(degree)) + ' degrees northwest' #左边
        return loc

def read_txt_as_int_list(file_path):
    with open(file_path, 'r') as file:
        int_list = [int(line.strip()) for line in file]
    return int_list

def face1(alpha):
    degree = alpha
    ##################################
    if -((6 * math.pi) / 10) < degree < -((4 * math.pi) / 10):
        loc = 'The object has its back to me, and its front side faces north.'
        return loc
    ##################################
    elif -((9 * math.pi) / 10) < degree <= -((6 * math.pi) / 10):
        loc = 'The left side of the object is facing me, and its front side faces northwest.'
        return loc
    elif -math.pi <= degree <= -((9 * math.pi) / 10):
        loc = 'The left side of the object is facing me, and its front side faces west.'
        return loc
    elif ((9 * math.pi) / 10) < degree <= math.pi:
        loc = 'The left side of the object is facing me, and its front side faces west.'
        return loc
    elif ((6 * math.pi) / 10) < degree <= ((9 * math.pi) / 10):
        loc = 'The left side of the object is facing me, and its front side faces southwest.'
        return loc
    elif ((5 * math.pi) / 10) < degree <= ((6 * math.pi) / 10):
        loc = 'The front of the object is facing me, and its front side faces south.'
        return loc
    elif ((4 * math.pi) / 10) < degree <= ((5 * math.pi) / 10):
        loc = 'The front of the object is facing me, and its front side faces south.'
        return loc
    elif ((1 * math.pi) / 10) < degree <= ((4 * math.pi) / 10):
        loc = 'The right side of the object is facing me, and its front side faces southeast.'
        return loc
    elif 0 < degree <= ((1 * math.pi) / 10):
        loc = 'The right side of the object is facing me, and its front side faces east'
        return loc
    elif -((1 * math.pi) / 10) < degree <= 0:
        loc = 'The right side of the object is facing me, and its front side faces east'
        return loc
    elif -((4 * math.pi) / 10) <= degree <= -((1 * math.pi) / 10):
        loc = 'The right side of the object is facing me, and its front side faces northeast.'
        return loc
    else:
        loc = None
        return loc
def import_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_data = json.loads(line.strip())
            data.append(line_data)
    return data


with open('/Mono3DRefer.json') as ff:
    data = json.dump(ff)
test_check_list = read_txt_as_int_list('/test.txt')

count = 0
count_object = 0
check_list = []
new_all_data_info = []
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_BASE_URL'] = ''
max_attemp = 99
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
    base_url=os.environ.get('OPENAI_BASE_URL')
)
for index, i in enumerate(data):
        label_list = ast.literal_eval(i['label_2'])
        beta = label_list[3] - label_list[14]
        status = i['status']
        color = i['color']
        # if abs(beta) > 1:
        #     print(index)
        #     count += 1
        #     print(count)
        #     print(beta)
        #     check_list.append(index)
        postion = location(beta)
        height = round(label_list[8], 1)
        width = round(label_list[9], 1)
        length = round(label_list[10], 1)
        distance = round(label_list[13])
        label = label_list[0]
        face = face1(label_list[3])

        if face is not None:
            for ii in range(0, 5):
                attemps = 0
                # caonima = 'caonima'
                while attemps < max_attemp:
                    try:
                        response = client.chat.completions.create(
                          model="gpt-4o",
                          messages=[
                            {
                              "role": "system",
                              "content": ""
                            },
                            {
                              "role": "user",
                              "content": ""
                            }
                          ],
                          temperature=0.7,
                          max_tokens=128,
                          top_p=1
                        )
                        attemps = 100
                    except Exception as e:
                        attemps += 1
                new_data_dict = i.copy()
                new_data_dict['description'] = response.choices[0].message.content
                new_data_dict['ann_id'] = ii
                new_data_dict['ann_index_in_total'] = count
                new_data_dict['obejet_id'] = count_object
                count += 1
                new_all_data_info.append(new_data_dict)
                print(response.choices[0].message.content)
                print(count)

            count_object += 1
        else:
            continue

with open('Mono3DRefer/', 'w', encoding='utf-8') as f:
      json.dump(new_all_data_info, f, ensure_ascii=False, indent=4)