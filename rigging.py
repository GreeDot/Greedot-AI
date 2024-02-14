import cv2
import mediapipe as mp
import yaml
import numpy as np
import pandas as pd

def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점 (관절)
    c = np.array(c)  # 세 번째 점

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 4)  # 소수점 네 자리까지 반올림하여 반환


# MediaPipe pose 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils # 그리기 유틸리티

# 관절 이름으로 저장할 조인트 목록
joint_names = [
    'mouth_right','mouth_left',
    'right_wrist', 'left_wrist', 'left_shoulder', 'right_elbow', 
    'left_elbow', 'right_shoulder',
    'left_ankle', 'left_knee', 'left_hip', 'right_ankle', 
    'right_knee', 'right_hip', 'nose'
]

#for mixamo skeleton
joint_rename = {
    'right_wrist' :'RightHand' ,
    'left_wrist' : 'LeftHand',
    'left_shoulder' : 'LeftShoulder', 
    'right_shoulder': 'RightShoulder',
    'right_elbow': 'RightArm',
    'left_elbow':'LeftArm' ,
    'left_ankle':'LeftFoot',
    'right_ankle':'RightFoot',
    'left_knee':'LeftLeg',
    'right_knee':'RightLeg' ,
    'left_hip': 'LeftUpLeg' , 
    'right_hip':'RightUpLeg' ,
    'nose' : 'Head'
}


# 동영상 파일 열기
cap = cv2.VideoCapture('2swonDance.mp4')
frame_data = []  # 프레임별 데이터 저장

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    new_size = (400, 400)
    image_resized = cv2.resize(image, new_size)

    # BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frame_coordinates = {}
        for joint_name, joint_rename_value in joint_rename.items():
            idx = eval(f"mp_pose.PoseLandmark.{joint_name.upper()}.value")
            landmark = results.pose_landmarks.landmark[idx]
            # 여기서는 이미지가 400x400으로 조정되었으므로, 좌표 조정은 필요 없음
            frame_coordinates[joint_rename_value] = (landmark.x * new_size[0], landmark.y * new_size[1], landmark.z)
        
        # 추가된 Neck, Hips, Spine2 좌표 계산
        frame_coordinates['Neck'] = tuple(np.mean([frame_coordinates['LeftShoulder'], frame_coordinates['RightShoulder']], axis=0))
        frame_coordinates['Hips'] = tuple(np.mean([frame_coordinates['LeftUpLeg'], frame_coordinates['RightUpLeg']], axis=0))
        frame_coordinates['Spine'] = frame_coordinates['Hips']
        frame_coordinates['Spine2'] = tuple(np.mean([frame_coordinates['LeftShoulder'], frame_coordinates['RightShoulder']], axis=0))

        frame_data.append(frame_coordinates)

pose.close()
cap.release()
# DataFrame으로 변환
df = pd.DataFrame(frame_data)

# 각도 계산 및 저장
angle_data = []
for frame in frame_data:
    angles = {
        'Spine': calculate_angle(frame['Spine2'], frame['Spine'], frame['Hips']),
        'LeftUpLeg': calculate_angle(frame['LeftLeg'], frame['LeftUpLeg'], frame['Hips']),
        'RightUpLeg': calculate_angle(frame['RightLeg'], frame['RightUpLeg'], frame['Hips']),
        'Spine2': calculate_angle(frame['Neck'], frame['Spine2'], frame['Spine']),
        'Neck': calculate_angle(frame['Head'], frame['Neck'], frame['Spine2']),
        'RightArm': calculate_angle(frame['RightHand'], frame['RightArm'], frame['RightShoulder']),
        'RightShoulder': calculate_angle(frame['RightArm'], frame['RightShoulder'], frame['Spine2']),
        'LeftHand': calculate_angle(frame['LeftHand'], frame['LeftArm'], frame['LeftShoulder']),
        'LeftShoulder': calculate_angle(frame['LeftArm'], frame['LeftShoulder'], frame['Spine2']),
        'RightLeg': calculate_angle(frame['RightFoot'], frame['RightLeg'], frame['RightUpLeg']),
        'LeftLeg': calculate_angle(frame['LeftFoot'], frame['LeftLeg'], frame['LeftUpLeg'])
    }
    angle_data.append(angles)

df_angles = pd.DataFrame(angle_data)

# CSV 파일로 저장
df.to_csv('pose_coordinates_combined.csv', index=False)
df_angles.to_csv('pose_angle_combined.csv', index=False)

joints = [
    "Hips","Spine","Spine1","Spine2",
    "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
    "LeftHandIndex1","LeftHandIndex2","LeftHandIndex3","LeftHandMiddle1",
    "LeftHandMiddle2","LeftHandMiddle3","LeftHandPinky1","LeftHandPinky2",
    "LeftHandPinky3","LeftHandRing1","LeftHandRing2",
    "LeftHandRing3","LeftHandThumb1","LeftHandThumb2",
    "LeftHandThumb3","Neck","Head","RightShoulder","RightArm",
    "RightForeArm","RightHand","RightHandIndex1",
    "RightHandIndex2","RightHandIndex3","RightHandMiddle1",
    "RightHandMiddle2","RightHandMiddle3","RightHandPinky1",
    "RightHandPinky2","RightHandPinky3","RightHandRing1",
    "RightHandRing2","RightHandRing3","RightHandThumb1",
    "RightHandThumb2","RightHandThumb3","LeftUpLeg","LeftLeg",
    "LeftFoot","LeftToeBase","RightUpLeg","RightLeg","RightFoot",
    "RightToeBase"
]


with open('Animation/my.bvh', 'a', encoding='utf-8') as file:
    file.write('\nMOTION\nFrames: {}\nFrame Time: 0.033333\n'.format(len(df)))

    # 각 프레임에 대한 데이터를 파일에 기록합니다.
    for frameNum in range(len(df)):
        frameAni = ""
        for joint in joints:
            if joint == 'Hips':
                hips_coordinates = df['Hips'].iloc[frameNum]  # (x, y, z) 형태의 튜플
                rounded_hips_coordinates = tuple(round(coord, 4) for coord in hips_coordinates)
                frameAni += "{} {} {} ".format(*rounded_hips_coordinates)
            if joint in df_angles.columns:
                frameAni += "0.0000 0.0000 " + str(df_angles[joint].iloc[frameNum]) + " "
            else:
                # 일치하는 관절이 없는 경우 "0.0000 0.0000 0.0000"을 추가합니다.
                frameAni += "0.0000 0.0000 0.0000 "
        
        # 각 프레임의 데이터를 파일에 작성합니다. 마지막에 개행 문자를 추가하여 프레임을 구분합니다.
        file.write(frameAni.strip() + '\n')
