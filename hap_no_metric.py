import cv2
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict
# from detect_shaft import DETECT_SHAFT
from detect_target import DETECT_TARGET
# from detect_target_2 import DETECT_TARGET_2
from visualize import TargetVisualizer
from collections import defaultdict
from ultralytics import YOLO
import argparse
import re
from scipy.spatial import distance
from scipy.spatial.distance import cdist


class ArcheryPoseEstimator:
    def __init__(self, model_path, source, source1, perspective_file, output_dir):
        """
        모델 및 데이터 경로 설정
        """
        self.model = YOLO(model_path)
        self.source = source
        self.source1 = source1
        self.perspective_file = perspective_file
        self.output_dir = output_dir
        self.cnt = 0  # 키포인트 트래킹을 위한 카운터
        self.tracked_keypoints = None  # 이전 프레임 키포인트 저장 (NumPy 배열)
        self.prev_detection_count = None  # 이전 프레임 감지된 객체 개수
        # 결과 저장 폴더 생성
        os.makedirs(self.output_dir, exist_ok=True)

        # Perspective 좌표 로드
        self.perspective_data = self.load_perspective_data()

    def load_perspective_data(self):
        """
        원근 변환 좌표 로드
        """
        perspective_data = {}
        with open(self.perspective_file, 'r') as f:
            for line in f:
                parts = line.strip().split(",")
                name, coords = parts[0], list(map(float, parts[1:]))
                perspective_data[name] = np.array(coords, dtype=np.float32).reshape(4, 2)
        return perspective_data

    def apply_perspective_transform(self, keypoints, M):
        """
        원근 변환 행렬 M을 사용하여 키포인트 변환
        """
        if keypoints is None or len(keypoints) == 0:
            return None

        keypoints = keypoints.reshape(-1, 2)  # (N, 2) 형태로 변환
        keypoints_homo = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])  # (x, y) → (x, y, 1)

        transformed_keypoints = np.dot(M, keypoints_homo.T).T  # 원근 변환 적용
        transformed_keypoints /= transformed_keypoints[:, 2].reshape(-1, 1)  # 정규화 (z=1)
        return transformed_keypoints[:, :2]  # (N, 2) 반환

    def indexing(self, transformed_keypoints, threshold=30):
        """
        거리 기반 트래킹 + 인덱싱
        - 이전 좌표와 비교하여 가장 가까운 좌표를 같은 것으로 간주
        - 새로운 좌표는 리스트에 추가하여 최신 상태 유지
        - 마지막 프레임의 최신 좌표를 self.tracked_keypoints에서 확인 가능
        - threshold: 같은 좌표로 판단할 거리 기준 (픽셀 단위)
        """
        transformed_keypoints = np.array(transformed_keypoints).reshape(-1, 2)  # (N, 2) 형태로 변환

        if self.tracked_keypoints is None:
            self.tracked_keypoints = transformed_keypoints  # 첫 좌표 저장
            return self.tracked_keypoints[-1]  # 최신 좌표 반환

        prev_keypoints = np.array(self.tracked_keypoints)  # (M, 2)

        distances = cdist(transformed_keypoints, prev_keypoints)  # 거리 행렬 계산
        matched_indices = {}

        for i, row in enumerate(distances):
            min_dist = np.min(row)
            min_idx = np.argmin(row)
            if min_dist < threshold:
                matched_indices[i] = min_idx

        updated_keypoints = np.zeros_like(prev_keypoints)
        updated_mask = np.zeros(prev_keypoints.shape[0], dtype=bool)

        for curr_idx, prev_idx in matched_indices.items():
            updated_keypoints[prev_idx] = transformed_keypoints[curr_idx]
            updated_mask[prev_idx] = True

        new_keypoints = transformed_keypoints[
            ~np.isin(range(len(transformed_keypoints)), list(matched_indices.keys()))
        ]
        self.tracked_keypoints = np.vstack([updated_keypoints[updated_mask], new_keypoints])

        print(f"마지막 좌표 : {self.tracked_keypoints[-1]}")
        return self.tracked_keypoints[-1]  # 최신 좌표 반환

    def process_images(self):
        """
        YOLO 모델을 이용해 이미지 예측 후, 원근 변환 적용 및 저장
        """
        # 이미지 파일 목록 정렬
        image_files_cam1 = [os.path.join(self.source, f) for f in sorted(os.listdir(self.source)) if f.endswith(('.png', '.jpg'))]
        image_files_cam3 = [os.path.join(self.source1, f) for f in sorted(os.listdir(self.source1)) if f.endswith(('.png', '.jpg'))]

        # YOLO 예측 실행
        results_cam1 = self.model.predict(image_files_cam1, batch=16, device="cuda", save=True, line_width=1, project=self.output_dir)
        results_cam3 = self.model.predict(image_files_cam3, batch=16, device="cuda", save=True, line_width=1, project=self.output_dir)

        for img_path_cam1, result_cam1, img_path_cam3, result_cam3 in zip(image_files_cam1, results_cam1, image_files_cam3, results_cam3):
            img_name_cam1 = os.path.basename(img_path_cam1)
            img_name_cam3 = os.path.basename(img_path_cam3)

            img_cam1 = cv2.imread(img_path_cam1)
            img_cam3 = cv2.imread(img_path_cam3)

            keypoints_cam1 = result_cam1.keypoints.xy.cpu().numpy() if result_cam1.keypoints is not None else None
            keypoints_cam3 = result_cam3.keypoints.xy.cpu().numpy() if result_cam3.keypoints is not None else None

            # 감지된 개수 계산
            detection_count_cam1 = len(keypoints_cam1) if keypoints_cam1 is not None else 0
            detection_count_cam3 = len(keypoints_cam3) if keypoints_cam3 is not None else 0

            # CAM1에서 새로운 객체가 감지되지 않았지만 CAM3에서 감지됨 → CAM3 사용
            if self.prev_detection_count == detection_count_cam1 and detection_count_cam3 > detection_count_cam1:
                print(f"{img_name_cam1}: CAM1에서 새로운 화살 감지 안됨 → CAM3 결과 사용")
                selected_keypoints = keypoints_cam3
                selected_img = img_cam3
                selected_img_name = img_name_cam3

            # CAM1에서 감지된 경우 → CAM1 결과 사용
            elif keypoints_cam1 is not None and keypoints_cam1.size > 0:
                print(f"{img_name_cam1}: CAM1에서 객체 검출됨")
                selected_keypoints = keypoints_cam1
                selected_img = img_cam1
                selected_img_name = img_name_cam1

            # CAM1과 CAM3 모두 검출되지 않은 경우 → 스킵
            else:
                print(f"{img_name_cam1}: CAM1, CAM3 모두에서 객체 검출 안됨 → 건너뜀")
                continue
            # 파일명에서 `_frame_숫자.확장자` 제거하여 기준 이름 추출
            base_name = re.sub(r"_frame_\d+\.(png|jpg)$", "", selected_img_name)
            # perspective_data에 해당 이름이 있는 경우 원근 변환 적용
            if base_name in self.perspective_data.keys():
                perspective_coords = self.perspective_data[base_name]

                # 1920×1920 해상도로 원근 변환
                dst_points = np.float32([
                    [0, 0], [1919, 0], [1919, 1919], [0, 1919]
                ])
                M = cv2.getPerspectiveTransform(perspective_coords, dst_points)

                # 이미지 원근 변환 적용 (1920×1920)
                transformed_img = cv2.warpPerspective(selected_img, M, (1920, 1920))

                # 키포인트에 원근 변환 적용
                transformed_keypoints = self.apply_perspective_transform(selected_keypoints, M)
                latest_keypoint = self.indexing(transformed_keypoints)
            

                # 변환된 키포인트에서 x, y 좌표 추출
                x, y = latest_keypoint
                #=======================================================================================================
                # # TODO: x,y 좌표가 마지막 좌표로 잘 됐는 지 디버깅
                # # 🔹 키포인트 시각화 (디버깅용)
                # for i, (px, py) in enumerate(transformed_keypoints):
                #     cv2.circle(transformed_img, (int(px), int(py)), 5, (0, 255, 0), -1)  # 초록색 (모든 변환된 키포인트)
                #     cv2.putText(transformed_img, f"{i}", (int(px) + 5, int(py) - 5), 
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)  # 번호 표시

                # # 🔹 최종 선택된 키포인트 시각화 (빨간색)
                # cv2.circle(transformed_img, (int(x), int(y)), 8, (0, 0, 255), -1)  # 빨간색 (최종 선택된 키포인트)
                # cv2.putText(transformed_img, "Final", (int(x) + 10, int(y) - 10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)  # "Final" 텍스트 추가

                # # 기존 tracked_keypoints 시각화 (파란색)
                # if self.tracked_keypoints is not None:
                #     for i, (tx, ty) in enumerate(self.tracked_keypoints):
                #         cv2.circle(transformed_img, (int(tx), int(ty)), 4, (255, 0, 0), -1)  # 파란색 (기존 트래킹된 키포인트)
                #         cv2.putText(transformed_img, f"T{i}", (int(tx) + 5, int(ty) - 5),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)  # "T" + 번호

                # # 이미지 저장 (디버깅용)
                debug_img_path = os.path.join(self.output_dir, f"debug_{selected_img_name}")
                cv2.imwrite(debug_img_path, transformed_img)
                print(f"✅ Debug Image Saved: {debug_img_path}")

                # # 변환된 이미지 저장
                bg_image_path = os.path.join(self.output_dir, f"perspective_{selected_img_name}")
                cv2.imwrite(bg_image_path, transformed_img)
                print(f"{bg_image_path} 저장 완료!")
                # =======================================================================================================
                
                
                # print(transformed_keypoints)
                hits = []
                # score_target = []
                for shaft_coord in transformed_keypoints:
                    x, y = shaft_coord  

                    # 각 화살의 위치별로 DETECT_TARGET을 실행하여 점수를 계산
                    target_detector = DETECT_TARGET(
                        bg_image_path,
                        int(x),
                        int(y),
                        min_area=5000,
                        max_area=1000000,
                        center_tolerance=300,
                        max_ellipses=15,
                    )       
                    center, score, merged_ellipses = target_detector.process_target_detection()

                    hits.append(
                        {
                            "point": (int(x), int(y)),
                            "score": score,  # 각 좌표별 개별적인 score 적용
                        }
                    )
                    


                # # Create the visualizer
                visualizer = TargetVisualizer(center[0], center[1])
                print(center)

                # # Example hit points with scores
                # hits = [{"point": (x, y), "score": score}]

                # Draw the visualization
                output_img = visualizer.visualize(transformed_img, hits)
                output_path = os.path.join(self.output_dir, f"score4_{selected_img_name}")
                cv2.imwrite(output_path, transformed_img)

                # Display the image
                cv2.imshow("Visualization", output_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            self.prev_detection_count = detection_count_cam1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='./pose_s_add_best.pt')
    parser.add_argument("--source", type=str, default='../labeling/archery_20250116/20250116_091103/cam1_4set/')
    parser.add_argument("--source1", type=str, default='../labeling/archery_20250116/20250116_091103/cam2_4set/')
    parser.add_argument("--perspective", type=str, default='../labeling/archery_20250116/20250116.txt')
    parser.add_argument("--output", type=str, default='./output_results')

    args = parser.parse_args()

    estimator = ArcheryPoseEstimator(
        model_path=args.model,
        source=args.source,
        source1=args.source1,
        perspective_file=args.perspective,
        output_dir=args.output
    )

    estimator.process_images()

    
