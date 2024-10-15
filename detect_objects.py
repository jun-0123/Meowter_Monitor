import torch
import cv2


class Model:
    def __init__(self, video_path):
        # YOLOv5 모델 불러오기
        print("모델을 불러오는 중...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("모델 불러오기 완료.")
        self.video_path = video_path

    def detect_objects(self):
        # 영상 캡처 객체 생성
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

            # 객체 탐지 실행
            results = self.model(frame)
            print("객체 탐지 완료.")

            # 탐지된 객체 중 고양이와 물그릇 식별
            for *xyxy, conf, cls in results.xyxy[0]:
                if results.names[int(cls)] in ['cat', 'bowl']:  # 고양이와 물그릇 탐지
                    print(f"감지된 객체: {results.names[int(cls)]}, 신뢰도: {conf:.2f}")

            # 탐지 결과를 이미지에 그리기
            cv2.imshow('YOLOv5s Object Detection', results.render()[0])
            print("탐지 결과를 이미지에 그리는 중...")

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q'를 눌러 종료합니다.")
                break

        cap.release()
        cv2.destroyAllWindows()

# 예시 사용법
video_path = './video/cat4.mp4'
model_instance = Model(video_path)
model_instance.detect_objects()
