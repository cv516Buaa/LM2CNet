import numpy as np
import json
import ast

def ry2alpha(cu, fu, ry, u):
    alpha = ry - np.arctan2(u - cu, fu)

    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi

    return alpha

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}



with open('trifical_all_data_info.json', 'r') as f:
    data = json.load(f)
for i in data:
    label_list = ast.literal_eval(i['label_2'])
    # name = i['im_name']
    calib_path = '/calib/{}/'.format(i['CAM']) + '{}.txt'.format(i['key_frames'])
    calib = get_calib_from_file(calib_path)
    p2 = calib['P2']
    u = (float(label_list[4]) + float(label_list[6])) / 2
    ry = float(label_list[14])
    fu = p2[0, 0]
    cu = p2[0, 2]
    alpha = ry2alpha(cu, fu, ry, u)
    label_list[3] = round(alpha, 2)
    b = str(label_list)
    i['label_2'] = b
with open('all_data_info.json', 'w', encoding='utf-8') as ff:
    json.dump(data, ff, ensure_ascii=False, indent=4)





