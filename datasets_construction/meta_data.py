import json
import os
from typing import List, Dict, Any
import tqdm
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs
def ry2alpha(cu, fu, ry, u):
    alpha = ry - np.arctan2(u - cu, fu)

    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi

    return alpha


def find_max_iou_index(dict_list):
    """根据字典中的'IOU'键的值，返回最大IOU值对应的列表索引"""
    max_iou = float('-inf')  # 初始化最大IOU值为负无穷
    max_index = -1  # 初始化最大值索引为-1

    for index, dictionary in enumerate(dict_list):
        if 'IOU' in dictionary:
            current_iou = dictionary['IOU']
            if current_iou > max_iou:
                max_iou = current_iou
                max_index = index

    return max_index
def calculate_iou(box1, box2):

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


##########################################
camer_check_dict = {
'CAM_FRONT': np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
'CAM_FRONT_LEFT': np.array([[1, -1, 0], [0, 0, -1], [1, 1, 0]]),
'CAM_FRONT_RIGHT': np.array([[-1, -1, 0], [0, 0, -1], [1, -1, 0]]),
'CAM_BACK': np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
'CAM_BACK_LEFT': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
'CAM_BACK_RIGHT': np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
}
imsize = (1600, 900)


###########################################
count = 0
count_image = 0
count_obejct = 0
myset = set()
wrong_data = []
all = {'Cycle', 'Vehicle', 'Pedestrian', 'Others', 'Traffic element'}
checked = ['Traffic element']
all_data_info = []
kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
label_folder = os.path.join('/media/3dvg_nus', 'trainval', 'label_2')
calib_folder = os.path.join('/media/3dvg_nus', 'trainval', 'calib')
# image_folder = os.path.join('/media/3dvg_nus', 'trainval', 'image_2')
lidar_folder = os.path.join('/media/3dvg_nus', 'trainval', 'velodyne')
with open('/media/v1_0_train_nus.json', 'r') as f:
    data = json.load(f)
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/nuscenes_full', verbose=True)
for i in data:
    for ii in data[i]['key_frames']:
        for iii in data[i]['key_frames'][ii]['key_object_infos']:
            if data[i]['key_frames'][ii]['key_object_infos'][iii]['Category'] in checked:
                ############################
                camer_check_list = iii.split(',')
                carmer = camer_check_list[1]
                image_name = data[i]['key_frames'][ii]['image_paths']['{}'.format(carmer)]
                categoty = data[i]['key_frames'][ii]['key_object_infos'][iii]['Category']
                status = data[i]['key_frames'][ii]['key_object_infos'][iii]['Status']
                visual_description = data[i]['key_frames'][ii]['key_object_infos'][iii]['Visual_description']
                bbox = data[i]['key_frames'][ii]['key_object_infos'][iii]['2d_bbox']
                ############################参数准备
                sample = nusc.get('sample', ii)
                sample_annotation_tokens = sample['anns']
                cam_front_token = sample['data'][carmer]
                lidar_token = sample['data']['LIDAR_TOP']
                ############################参数获取
                sd_record_cam = nusc.get('sample_data', cam_front_token)
                sd_record_lid = nusc.get('sample_data', lidar_token)
                cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
                ############################参数处理
                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                              inverse=False)
                ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                              inverse=True)
                velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
                # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
                velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)
                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
                r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.
                # Projection matrix.
                p_left_kitti = np.zeros((3, 4))
                p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.
                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]
                assert (velo_to_cam_rot.round(0) == camer_check_dict['{}'.format(carmer)]).all()  # BACK_RIGHT
                assert (velo_to_cam_trans[1:3] < 0).all()
                ############################初始化
                kitti_transforms = dict()
                kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
                kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
                kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
                calib_path = calib_folder + '/{}/'.format(carmer) + ii + '.txt'
                directory = os.path.dirname(calib_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                with open(calib_path, "w") as calib_file:
                    for (key, val) in kitti_transforms.items():
                        val = val.flatten()
                        val_str = '%.12e' % val[0]
                        for v in val[1:]:
                            val_str += ' %.12e' % v
                        calib_file.write('%s: %s\n' % (key, val_str))
                ###########################################
                anno_dict_check = []
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
                    # sample_attr = self.nusc.get('attribute', sample_annotation['attribute_tokens'])
                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                     selected_anntokens=[sample_annotation_token])
                    box_lidar_nusc = box_lidar_nusc[0]
                    attr = sample_annotation['attribute_tokens']
                    if len(attr) == 0:
                        describe = '-1'
                    else:
                        attribute = nusc.get('attribute', attr[0])
                        describe = attribute['name']

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    # occluded = 0
                    occluded = int(sample_annotation['visibility_token'])

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])

                    # Skip categories that are not part of the nuScenes detection challenge.
                    if detection_name is None:
                        continue

                    # Convert from nuScenes to KITTI box format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                    if bbox_2d is None:
                        continue
                    else:
                        iou_score = calculate_iou(bbox_2d, bbox)
                        if iou_score > 0.5:                   # Set dummy score so we can use this file as result.
                            box_cam_kitti.score = 0
                            v = np.dot(box_cam_kitti.rotation_matrix, np.array([1, 0, 0]))
                            yaw = -np.arctan2(v[2], v[0])
                            p2 = kitti_transforms['P2'].reshape(3, 4)
                            u = (bbox_2d[0] + bbox_2d[1]) / 2
                            ry = yaw
                            fu = p2[0, 0]
                            cu = p2[0, 2]
                            alpha = ry2alpha(cu, fu, ry, u)
                            label_2 = ['{}'.format(categoty), 0.0, 0.0, round(alpha, 2), round(bbox_2d[0], 2), round(bbox_2d[1], 2), round(bbox_2d[2], 2), round(bbox_2d[3], 2),
                                       round(box_cam_kitti.wlh[2], 2), round(box_cam_kitti.wlh[0], 2), round(box_cam_kitti.wlh[1], 2),
                                       round(box_cam_kitti.center[0], 2), round(box_cam_kitti.center[1], 2), round(box_cam_kitti.center[2], 2), round(yaw, 2)]
                            anno_dict = {
                            'IOU':iou_score,
                            'image_file_name': image_name[2:],
                            'instance_id': count,
                            'imname': image_name[2:],
                            'CAM': carmer,
                            'scene_token': i,
                            'key_frames': ii,
                            'objectName': categoty,
                            'kitti_class': detection_name,
                            'status': status,
                            'color': visual_description,
                            'ann_id': 0,
                            'description': '',
                            'label_2': str(label_2)
                            }
                            anno_dict_check.append(anno_dict)
                            # Convert box to output string format.
                            # output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                            #                                truncation=truncated, occlusion=occluded, attr=describe)

                            # # Write to disk.
                            # label_file.write(output + '\n')
                            break
                        else:
                            continue
                if len(anno_dict_check) == 1:
                    all_data_info.append(anno_dict_check[0])
                    anno_dict_check = []
                    count += 1
                    print(count)
                elif len(anno_dict_check) == 0:
                    wrong_data_dict = {
                        'scene_token': i,
                        'key_frames': ii,
                        'obejct': iii
                    }
                    wrong_data.append(wrong_data_dict)
                else:
                    index = find_max_iou_index(anno_dict_check)
                    all_data_info.append(anno_dict_check[index])
                    anno_dict_check = []
                    count += 1
                    print(count)
            else:
                continue

with open('/media/all_data_info.json', 'w', encoding='utf-8') as f:
    json.dump(all_data_info, f, ensure_ascii=False, indent=4)
###########################################
