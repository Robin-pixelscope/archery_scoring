import cv2
import os
import numpy as np
import time
import re
import argparse
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict
from ultralytics import YOLO
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scoring_module.detect_target import DETECT_TARGET
from scoring_module.assign_score import ASSIGN_SCORE
from scoring_module.visualize import TargetVisualizer


class ArcheryPoseEstimator:
    def __init__(self, model_path, perspective_file, output_dir):
        self.model = YOLO(model_path)
        # self.source = source
        # self.source1 = source1
        self.perspective_file = perspective_file
        self.output_dir = output_dir
        self.cnt = 0
        self.tracked_keypoints = None
        self.prev_detection_count = 0
        os.makedirs(self.output_dir, exist_ok=True)
        self.perspective_data = self.load_perspective_data()

    def load_perspective_data(self):
        perspective_data = {}
        with open(self.perspective_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                name, coords = parts[0], list(map(float, parts[1:]))
                perspective_data[name] = np.array(coords, dtype=np.float32).reshape(
                    4, 2
                )
        return perspective_data

    def apply_perspective_transform(self, keypoints, M):
        if keypoints is None or len(keypoints) == 0:
            return None
        keypoints = keypoints.reshape(-1, 2)
        keypoints_homo = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        transformed_keypoints = np.dot(M, keypoints_homo.T).T
        transformed_keypoints /= transformed_keypoints[:, 2].reshape(-1, 1)
        return transformed_keypoints[:, :2]

    def indexing(self, transformed_keypoints, threshold=30, movement_threshold=30):
        """
        거리 기반 트래킹 + 변화 감지로 새로운 화살 판단
        - threshold: 기존 좌표와의 매칭을 위한 거리 기준
        - movement_threshold: 좌표 변화량을 기반으로 새로운 화살 판단
        """
        transformed_keypoints = np.array(transformed_keypoints).reshape(-1, 2)

        # 첫 프레임일 경우 바로 저장
        if self.tracked_keypoints is None or self.tracked_keypoints.size == 0:
            self.tracked_keypoints = transformed_keypoints
            return self.tracked_keypoints[-1]

        prev_keypoints = np.array(self.tracked_keypoints)
        distances = cdist(transformed_keypoints, prev_keypoints)
        matched_indices = {}

        # 매칭된 좌표 기록
        for i, row in enumerate(distances):
            min_dist = np.min(row)
            min_idx = np.argmin(row)

            # 1. 기존 좌표와 가까운 경우 매칭
            if min_dist < threshold:
                matched_indices[i] = min_idx

        updated_keypoints = np.zeros_like(prev_keypoints)
        updated_mask = np.zeros(prev_keypoints.shape[0], dtype=bool)

        # 기존 키포인트 업데이트
        for curr_idx, prev_idx in matched_indices.items():
            movement = np.linalg.norm(
                transformed_keypoints[curr_idx] - prev_keypoints[prev_idx]
            )

            # 2. 이동 거리가 너무 크면 새로운 화살로 간주
            if movement > movement_threshold:
                print(
                    f"이전: {prev_keypoints[prev_idx]}, 현재: {transformed_keypoints[curr_idx]}"
                )
                self.tracked_keypoints = np.vstack(
                    [self.tracked_keypoints, transformed_keypoints[curr_idx]]
                )
            else:
                updated_keypoints[prev_idx] = transformed_keypoints[curr_idx]
                updated_mask[prev_idx] = True

        # 매칭되지 않은 새 키포인트 추가
        new_keypoints = transformed_keypoints[
            ~np.isin(range(len(transformed_keypoints)), list(matched_indices.keys()))
        ]
        self.tracked_keypoints = np.vstack(
            [updated_keypoints[updated_mask], new_keypoints]
        )

        if self.tracked_keypoints.size == 0:
            return None  # 빈 상태면 None 반환
        return self.tracked_keypoints[-1]

    def calibrate_missing_keypoints(
        self,
        transformed_keypoints,
        intrinsic_src,
        extrinsic_src,
        intrinsic_dst,
        extrinsic_dst,
    ):
        calibrated_keypoints = []

        # Source 카메라의 Extrinsic 분리
        R_src, T_src = extrinsic_src
        R_dst, T_dst = extrinsic_dst

        # 역행렬 계산
        K_src_inv = np.linalg.inv(intrinsic_src)
        R_src_inv = np.linalg.inv(R_src)

        for point in transformed_keypoints:
            # 1. 2D → 3D로 복원 (카메라 좌표계 → 월드 좌표계)
            point_homogeneous = np.append(point, 1)
            cam_coords = K_src_inv @ point_homogeneous

            world_coords = R_src_inv @ (cam_coords - T_src)

            # 2. 월드 좌표 → 대상 카메라 좌표계
            dst_cam_coords = R_dst @ world_coords + T_dst

            # 3. 3D → 2D로 투영
            dst_point_homogeneous = intrinsic_dst @ dst_cam_coords
            dst_point = dst_point_homogeneous[:2] / dst_point_homogeneous[2]

            calibrated_keypoints.append(dst_point)

        return np.array(calibrated_keypoints)

    def save_transformed_image(self, image, image_name):
        """
        변환된 이미지를 지정된 디렉토리에 저장합니다.

        :param image: 변환된 이미지
        :param image_name: 저장할 이미지의 이름
        """
        save_path = os.path.join(self.output_dir, f"perspective_{image_name}")
        cv2.imwrite(save_path, image)
        print(f"{save_path} 저장 완료!")
        return save_path

    def process_images(self, img_path_cam1, img_path_cam3):
        result_cam1 = self.model.predict(
            [img_path_cam1],
            batch=16,
            device="cuda",
            save=True,
            line_width=1,
            project=self.output_dir,
        )[0]
        result_cam3 = self.model.predict(
            [img_path_cam3],
            batch=16,
            device="cuda",
            save=True,
            line_width=1,
            project=self.output_dir,
        )[0]
        # exit()

        # for img_path_cam1, result_cam1, img_path_cam3, result_cam3 in zip(image_files_cam1, results_cam1, image_files_cam3, results_cam3):
        keypoints_cam1 = (
            result_cam1.keypoints.xy.cpu().numpy()
            if result_cam1.keypoints is not None
            else np.empty((0, 2))
        )
        keypoints_cam3 = (
            result_cam3.keypoints.xy.cpu().numpy()
            if result_cam3.keypoints is not None
            else np.empty((0, 2))
        )

        base_name_cam1 = re.sub(
            r"_frame_\d+\.(png|jpg)$", "", os.path.basename(img_path_cam1)
        )
        base_name_cam3 = re.sub(
            r"_frame_\d+\.(png|jpg)$", "", os.path.basename(img_path_cam3)
        )
        transformed_keypoints_cam1 = transformed_keypoints_cam3 = None

        # CAM1 원근 변환
        if keypoints_cam1.size > 0 and base_name_cam1 in self.perspective_data:
            M1 = cv2.getPerspectiveTransform(
                self.perspective_data[base_name_cam1],
                np.float32([[0, 0], [1919, 0], [1919, 1919], [0, 1919]]),
            )
            transformed_img = cv2.warpPerspective(
                cv2.imread(img_path_cam1), M1, (1920, 1920)
            )
            transformed_keypoints_cam1 = self.apply_perspective_transform(
                keypoints_cam1, M1
            )
            print(transformed_keypoints_cam1)
            selected_img_name = os.path.basename(img_path_cam1)

        # CAM3 원근 변환
        if keypoints_cam3.size > 0 and base_name_cam3 in self.perspective_data:
            M3 = cv2.getPerspectiveTransform(
                self.perspective_data[base_name_cam3],
                np.float32([[0, 0], [1919, 0], [1919, 1919], [0, 1919]]),
            )
            transformed_keypoints_cam3 = self.apply_perspective_transform(
                keypoints_cam3, M3
            )

            # CAM1이 없는 경우 CAM3로 transformed_img와 selected_img_name을 설정
            if (
                transformed_keypoints_cam1 is None
                or transformed_keypoints_cam1.size == 0
            ):
                transformed_img = cv2.warpPerspective(
                    cv2.imread(img_path_cam3), M3, (1920, 1920)
                )
                selected_img_name = os.path.basename(img_path_cam3)

        # CAM1과 CAM3 모두에서 검출된 게 없는 경우 → 건너뜀
        if (
            transformed_keypoints_cam1 is None or transformed_keypoints_cam1.size == 0
        ) and (
            transformed_keypoints_cam3 is None or transformed_keypoints_cam3.size == 0
        ):
            print(
                f"{base_name_cam1}: CAM1과 CAM3 모두에서 검출된 키포인트 없음 → 건너뜀"
            )
            return None, None, None, None

        # 현재 프레임의 검출 개수
        detection_count_cam1 = len(keypoints_cam1)
        detection_count_cam3 = len(keypoints_cam3)

        # 이전 프레임과 비교하여 변화 여부 판단
        cam1_changed = detection_count_cam1 > self.prev_detection_count
        cam3_changed = detection_count_cam3 > self.prev_detection_count

        # 한쪽에서만 검출 개수가 늘어난 경우, 캘리브레이션 진행
        if cam1_changed and not cam3_changed:
            # CAM3 보정
            transformed_keypoints_cam3 = self.calibrate_missing_keypoints(
                transformed_keypoints_cam1,
                intrinsic_cam1,
                extrinsic_cam1,
                intrinsic_cam3,
                extrinsic_cam3,
            )
            print(transformed_keypoints_cam3)
            print(f"{base_name_cam3}: CAM3은 CAM1로부터 보정 완료")

        elif cam3_changed and not cam1_changed:
            # CAM1 보정
            transformed_keypoints_cam1 = self.calibrate_missing_keypoints(
                transformed_keypoints_cam3,
                intrinsic_cam3,
                extrinsic_cam3,
                intrinsic_cam1,
                extrinsic_cam1,
            )
            print(transformed_keypoints_cam1)
            print(f"{base_name_cam1}: CAM1은 CAM3로부터 보정 완료")

        # CAM1을 기준으로 인덱싱
        if (
            transformed_keypoints_cam1 is not None
            and transformed_keypoints_cam1.size > 0
        ):
            latest_keypoint = self.indexing(transformed_keypoints_cam1)
            print(f"Latest Keypoint from CAM1: {latest_keypoint}")

        # 디버깅 출력
        print(f"Transformed CAM1 Keypoints: {transformed_keypoints_cam1}")
        print(f"Transformed CAM3 Keypoints: {transformed_keypoints_cam3}")

        # 현재 프레임의 검출 개수를 저장하여 다음 비교에 사용
        self.prev_detection_count = max(detection_count_cam1, detection_count_cam3)

        self.prev_detection_count = len(transformed_keypoints_cam1)

        # 이미지 저장
        if transformed_img is not None and selected_img_name is not None:
            bg_image_path = self.save_transformed_image(
                transformed_img, selected_img_name
            )

            return (
                bg_image_path,
                transformed_img,
                selected_img_name,
                transformed_keypoints_cam1,
            )

        return None, None, None, None


# Intrinsic Matrices
intrinsic_cam1 = np.array(
    [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]  # fx, 0, cx  # 0, fy, cy  # 0, 0, 1
)

intrinsic_cam3 = np.array(
    [[950, 0, 960], [0, 950, 540], [0, 0, 1]]  # fx, 0, cx  # 0, fy, cy  # 0, 0, 1
)

# Extrinsic Parameters
R1 = np.eye(3)  # CAM1: 회전 없음
T1 = np.array([0, 0, 0])  # CAM1: 이동 없음

# CAM3: 10도 회전 + 0.5m 이동
theta = np.radians(10)
R3 = np.array(
    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
)
T3 = np.array([0.5, 0, 0])  # 0.5m 오른쪽 이동

# Extrinsic을 튜플로 묶기
extrinsic_cam1 = (R1, T1)
extrinsic_cam3 = (R3, T3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./pose_s_add_best.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="../labeling/archery_20250116/20250116_094453/cam1_2set/",
    )
    parser.add_argument(
        "--source1",
        type=str,
        default="../labeling/archery_20250116/20250116_094453/cam2_2set/",
    )
    parser.add_argument(
        "--perspective", type=str, default="../labeling/archery_20250116/20250116.txt"
    )
    parser.add_argument("--output", type=str, default="./output_results")

    args = parser.parse_args()

    estimator = ArcheryPoseEstimator(
        model_path=args.model,
        # source=args.source,
        # source1=args.source1,
        perspective_file=args.perspective,
        output_dir=args.output,
    )

    cam1_images = sorted(
        [
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if f.endswith((".png", ".jpg"))
        ]
    )
    cam3_images = sorted(
        [
            os.path.join(args.source1, f)
            for f in os.listdir(args.source1)
            if f.endswith((".png", ".jpg"))
        ]
    )

    for cam1_img, cam3_img in zip(cam1_images, cam3_images):
        (
            bg_image_path,
            transformed_img,
            selected_img_name,
            transformed_keypoints_cam1,
        ) = estimator.process_images(cam1_img, cam3_img)

        hits = []
        if transformed_keypoints_cam1 is not None:
            # score_target = []
            for shaft_coord in transformed_keypoints_cam1:
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
                center, score, merged_ellipses = (
                    target_detector.process_target_detection()
                )

                hits.append(
                    {
                        "point": (int(x), int(y)),
                        "score": score,  # 각 좌표별 개별적인 score 적용
                    }
                )

            # Create the visualizer
            visualizer = TargetVisualizer(center[0], center[1])

            # # Example hit points with scores
            # hits = [{"point": (x, y), "score": score}]

            # Draw the visualization
            output_img = visualizer.visualize(transformed_img, hits)
            output_path = os.path.join(args.output, f"score4_{selected_img_name}")
            cv2.imwrite(output_path, transformed_img)

            # Display the image
            cv2.imshow("Visualization", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
