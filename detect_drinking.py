import torch
import cv2
import time

from detect_objects import Model


def detect_drinking(cap, model, threshold=2):
    fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 FPS 추출
    consecutive_frames = 0
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model.detect_objects(frame)
        cat_bbox = None
        bowl_bbox = None
        for detection in detections:
            if detection['label'] == 'cat':
                cat_bbox = detection['bbox']
            elif detection['label'] == 'bowl':
                bowl_bbox = detection['bbox']

        if cat_bbox and bowl_bbox:
            if (cat_bbox[0] < bowl_bbox[2] and cat_bbox[2] > bowl_bbox[0] and
                    cat_bbox[1] < bowl_bbox[3] and cat_bbox[3] > bowl_bbox[1]):
                if start_time is None:
                    start_time = time.time()  # 겹치기 시작한 시간 기록
                consecutive_frames += 1
            else:
                start_time = None
                consecutive_frames = 0

        if start_time is not None and (time.time() - start_time) > threshold:
            print("고양이가 물을 마시는 것이 3초 이상 감지됨!")
            break  # 또는 필요한 추가 처리 수행

    cap.release()
    cv2.destroyAllWindows()


video_path = './video/cat4.mp4'
# 객체 탐지 모델 초기화
model = Model(video_path)

# 비디오 캡처 초기화
cap = cv2.VideoCapture(video_path)

detect_drinking(cap, model)
