from datetime import datetime
import os
import time
import torch
import cv2


# Model 클래스 정의
class Model:
    def __init__(self, video_path):
        print("모델을 불러오는 중...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("모델 불러오기 완료.")
        self.video_path = video_path

    def detect_objects(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("영상을 열 수 없습니다. 비디오 경로를 확인해주세요:", self.video_path)
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("영상 읽기를 종료합니다.")
                break

            print("영상에서 프레임을 성공적으로 읽었습니다.")

            results = self.model(frame)
            print("객체 탐지 완료.")

            for *xyxy, conf, cls in results.xyxy[0]:
                if results.names[int(cls)] in ['cat', 'bowl']:
                    print(f"감지된 객체: {results.names[int(cls)]}, 신뢰도: {conf:.2f}")

            cv2.imshow('YOLOv5s Object Detection', results.render()[0])
            print("탐지 결과를 이미지에 그리는 중...")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q'를 눌러 종료합니다.")
                break

        cap.release()
        cv2.destroyAllWindows()


# CatDrinkingDetector 클래스 정의
class CatDrinkingDetector:
    def __init__(self, video_path):
        print("모델을 불러오는 중...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'best', pretrained=True)
        print("모델 불러오기 완료.")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        # 웹캠에서 실시간 영상을 받아오는 코드
        # self.cap = cv2.VideoCapture(0)  # '0'은 웹캠의 인덱스입니다.
        if not self.cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다. 비디오 경로를 확인해주세요: {video_path}")
        self.image_save_path = './image'
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            label = results.names[int(cls)]
            if label in ['cat', 'bowl']:
                detections.append({'label': label, 'bbox': xyxy, 'confidence': conf})
        return detections

    def detect_drinking(self, threshold=2):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        consecutive_frames = 0
        start_time = None
        frame_count = 0  # 프레임 카운트 추가
        overlap_time = 0  # 겹친 시간을 저장하는 변수

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detect_objects(frame)
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
                        start_time = time.time()
                    consecutive_frames += 1
                else:
                    if start_time is not None:
                        overlap_time = time.time() - start_time
                        print(f"고양이와 그릇이 겹쳐져 있었던 시간: {overlap_time:.2f}초")
                    start_time = None
                    consecutive_frames = 0

            if start_time is not None and (time.time() - start_time) > threshold:
                print("고양이가 물을 마시는 것이 감지됨!")
                # 현재 날짜와 시간을 포함한 파일 이름 생성
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_save_path = os.path.join(self.image_save_path, f"cat_drinking_{current_time}.jpg")
                cv2.imwrite(img_save_path, frame)
                print(f"이미지 저장됨: {img_save_path}")
                overlap_time = time.time() - start_time
                print(f"고양이와 그릇이 겹쳐져 있었던 총 시간: {overlap_time:.2f}초")
                break

            frame_count += 1  # 프레임 카운트 증가

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.detect_drinking()


# 메인 프로그램
if __name__ == "__main__":
    video_path = './video/my_cat2.mp4'

    # Model 인스턴스 생성 및 객체 탐지 실행
    model_instance = Model(video_path)
    model_instance.detect_objects()

    # CatDrinkingDetector 인스턴스 생성 및 실행
    detector = CatDrinkingDetector(video_path)
    detector.run()
