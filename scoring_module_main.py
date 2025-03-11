import cv2
import os
import numpy as np
import re
import json
import argparse
from collections import deque
from scipy.spatial.distance import cdist
from ultralytics import YOLO
from scoring_module.detect_target import DETECT_TARGET
from scoring_module.assign_score import ASSIGN_SCORE
from scoring_module.visualize import TargetVisualizer


# 1. 세트별 파일 읽기 클래스
class SetFileReader:
    def __init__(self, source_cam1, source_cam3):
        self.source_cam1 = source_cam1
        self.source_cam3 = source_cam3

    def get_image_files(self):
        files_cam1 = sorted(
            [
                os.path.join(self.source_cam1, f)
                for f in os.listdir(self.source_cam1)
                if f.lower().endswith((".png", ".jpg"))
            ]
        )
        files_cam3 = sorted(
            [
                os.path.join(self.source_cam3, f)
                for f in os.listdir(self.source_cam3)
                if f.lower().endswith((".png", ".jpg"))
            ]
        )
        return files_cam1, files_cam3

    def get_current_set_name(self, filename):
        # 파일 이름에서 세트 이름 추출 (예: "20250116_091103_cam1_1set")
        parts = filename.split("_")
        return "_".join(parts[:5])


# 2. 화살 좌표 추출 및 변환 이미지 저장 클래스
class PerspectiveTransformer:
    def __init__(self, perspective_file, output_dir):
        self.perspective_file = perspective_file
        self.output_dir = output_dir
        self.perspective_data = self.load_perspective_data()
        self.tracked_keypoints = None  # 화살 좌표 트래킹

    def load_perspective_data(self):
        perspective_data = {}
        with open(self.perspective_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                name = parts[0]
                coords = list(map(float, parts[1:]))
                perspective_data[name] = np.array(coords, dtype=np.float32).reshape(
                    4, 2
                )
        return perspective_data

    def apply_perspective_transform(self, image, keypoints, base_name):
        if base_name in self.perspective_data:
            perspective_coords = self.perspective_data[base_name]
            dst_points = np.float32([[0, 0], [1919, 0], [1919, 1919], [0, 1919]])
            M = cv2.getPerspectiveTransform(perspective_coords, dst_points)
            transformed_img = cv2.warpPerspective(image, M, (1920, 1920))
            transformed_keypoints = None
            if keypoints is not None and len(keypoints) > 0:
                transformed_keypoints = self._transform_keypoints(keypoints, M)
            return transformed_img, transformed_keypoints, M
        else:
            # perspective_data에 해당 이름이 없으면 원본 반환
            return image, keypoints, None

    def _transform_keypoints(self, keypoints, M):
        keypoints = keypoints.reshape(-1, 2)
        keypoints_homo = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
        transformed = np.dot(M, keypoints_homo.T).T
        transformed = transformed / transformed[:, 2].reshape(-1, 1)
        return transformed[:, :2]

    def index_keypoints(self, transformed_keypoints, threshold=30):
        transformed_keypoints = np.array(transformed_keypoints).reshape(-1, 2)
        if self.tracked_keypoints is None:
            self.tracked_keypoints = transformed_keypoints
            return self.tracked_keypoints[-1]
        prev_keypoints = np.array(self.tracked_keypoints)
        distances = cdist(transformed_keypoints, prev_keypoints)
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
        self.tracked_keypoints = np.vstack(
            [updated_keypoints[updated_mask], new_keypoints]
        )
        print(f"Latest tracked keypoint: {self.tracked_keypoints[-1]}")
        return self.tracked_keypoints[-1]

    def save_transformed_image(self, transformed_img, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"perspective_{filename}")
        cv2.imwrite(path, transformed_img)
        # print(f"Transformed image saved: {path}")
        return path


# 3. TargetDetector를 활용한 점수영역 컨투어 검출 클래스
class TargetDetector:
    def __init__(
        self,
        circularity_threshold=0.85,
        min_area=10000,
        max_area=10000000,
        min_length=500.0,
        max_length=20000.0,
        center_tolerance=300,
        max_contours_count=10,
    ):
        self.circularity_threshold = circularity_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_length = min_length
        self.max_length = max_length
        self.center_tolerance = center_tolerance
        self.max_contours_count = max_contours_count

    def get_contours(self, image_path):
        target_detector = DETECT_TARGET(
            image_path,
            circularity_threshold=self.circularity_threshold,  # 원형도 임계값
            min_area=self.min_area,  # 너무 작은 영역은 제거
            max_area=self.max_area,  # 너무 큰 영역은 제거
            min_length=self.min_length,  # 윤곽 길이(둘레)가 너무 짧은 것 제거
            max_length=self.max_length,  # 너무 긴 것도 제거
            center_tolerance=self.center_tolerance,  # 중심 좌표 허용 오차
            max_contours_count=self.max_contours_count,  # 최대 컨투어 개수
            # debug = False,  # 디버깅 모드
        )
        (cX_0, cY_0), contours_of_points = target_detector.process_target_detection()

        return (cX_0, cY_0), contours_of_points


# 4. Scorer를 활용한 점수 할당 클래스
class Scorer:
    def __init__(self):
        self.scoring = ASSIGN_SCORE()

    def get_score(self, center, arrow, contours_of_points):
        """
        중심 좌표와 화살 좌표, 그리고 점수별 컨투어(폴리곤)를 이용하여
        화살의 위치에 따른 세분화된 점수를 선형 보간 방식으로 할당하는 함수.

        Parameters:
            center (tuple): (cX_0, cY_0) 중심 좌표
            arrow (tuple): 화살 좌표 (shaft position)
            contours_of_points (list of ndarray): 점수별 컨투어(폴리곤) 목록
                                                (예: cv2.findContours 결과)

        Returns:
            score (float or None): 최종 할당 점수 (교차점 계산이 안되면 fallback으로 면적 기반 계산)
        """
        score = self.scoring.assign_score(center, arrow, contours_of_points)

        return score


# 5. 정확도 검증 및 결과 JSON 저장 클래스
class AccuracyValidator:
    def __init__(
        self,
        json_truth_file="./updated_truth_annotations.json",
        json_output_file="./upgrade_score_results.json",
    ):
        self.json_truth_file = json_truth_file
        self.json_output_file = json_output_file
        self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
        self.set_sum = 0
        self.prev_detected_count = 0
        self.existing_scores = self.load_existing_scores()

    def load_existing_scores(self):
        if os.path.exists(self.json_truth_file):
            with open(self.json_truth_file, "r") as f:
                return json.load(f)
        return {}

    def get_existing_score(self, img_name):
        return self.existing_scores.get(img_name, {}).get("score", 0)

    def update_metrics(self, img_name, detected_count, accumulated_score):
        truth_score = self.get_existing_score(img_name)
        detected_increase = detected_count - self.prev_detected_count
        # 1. 검출 성공 & 점수 정확
        if truth_score > 0 and detected_count > 0 and accumulated_score == truth_score:
            self.tp += 1
            print(f"✅ TP: {img_name} - 검출 성공 (점수 정확)")
        # 2. 실제 화살 있음 → 검출 안됨
        elif truth_score > self.set_sum and detected_count <= self.prev_detected_count:
            self.fn += 1
            print(f"❌ FN: {img_name} - 점수 있음({truth_score}) → 검출 안됨")
        # 3. 점수 없음인데 검출됨 (노이즈 가능)
        elif truth_score == 0 and detected_count > 0:
            self.fp += 1
            print(f"🚨 FP: {img_name} - 점수 없음인데 검출됨")
        # 4. 갑작스런 화살 개수 증가 혼합 경우
        elif detected_increase >= 2:
            if accumulated_score == truth_score:
                self.tp += 1
                print(f"✅ TP: {img_name} - 검출 성공 (점수 정확)")
            else:
                self.fp += 1
                print(f"🚨 FP: {img_name} - 갑작스런 화살 개수 증가 (노이즈 가능)")
        # 5. 검출됐지만 점수 불일치
        elif detected_count > 0 and accumulated_score != truth_score:
            if accumulated_score > truth_score:
                self.fp += 1
                print(f"🚨 FP: {img_name} - 검출됐으나 과검출")
            else:
                self.fn += 1
                print(f"❌ FN: {img_name} - 검출됐으나 미검출")
        # 6. 점수 없음 & 검출 안됨
        elif truth_score == 0 and detected_count == 0:
            self.tn += 1
            print(f"🟢 TN: {img_name} - 점수 없음, 검출도 안됨")
        self.prev_detected_count = detected_count
        self.set_sum = truth_score
        return truth_score

    def compute_metrics(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + 1e-6)
        return precision, recall, accuracy

    def save_results(self, img_name, truth_score, detected_count, accumulated_score):
        precision, recall, accuracy = self.compute_metrics()
        score_results = {
            "truth_score": truth_score,
            "detected_count": detected_count,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
        }
        if os.path.exists(self.json_output_file):
            try:
                with open(self.json_output_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}
        existing_data[img_name] = score_results
        with open(self.json_output_file, "w") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 결과가 {self.json_output_file} 파일에 저장되었습니다!")


# main 함수: 각 클래스를 활용하여 전체 파이프라인 실행
def main(args):
    # YOLO 모델 로드
    model = YOLO(args.model)

    # 클래스 인스턴스 생성
    file_reader = SetFileReader(args.source, args.source1)
    transformer = PerspectiveTransformer(args.perspective, args.output)
    target_detector = TargetDetector(
        circularity_threshold=0.85,
        min_area=10000,
        max_area=10000000,
        min_length=500.0,
        max_length=20000.0,
        center_tolerance=300,
        max_contours_count=10,
    )
    scorer = Scorer()
    validator = AccuracyValidator()

    image_files_cam1, image_files_cam3 = file_reader.get_image_files()

    # YOLO 예측 (CAM1, CAM3)
    results_cam1 = model.predict(
        image_files_cam1,
        batch=16,
        device="cuda",
        save=True,
        line_width=1,
        project=args.output,
    )
    results_cam3 = model.predict(
        image_files_cam3,
        batch=16,
        device="cuda",
        save=True,
        line_width=1,
        project=args.output,
    )

    contours_list = deque(maxlen=10)
    hits = []
    for img_path_cam1, result_cam1, img_path_cam3, result_cam3 in zip(
        image_files_cam1, results_cam1, image_files_cam3, results_cam3
    ):
        img_name_cam1 = os.path.basename(img_path_cam1)
        img_name_cam3 = os.path.basename(img_path_cam3)

        current_set = file_reader.get_current_set_name(img_name_cam1)
        print(f"현재 세트: {current_set}")

        # 이미지 로드
        img_cam1 = cv2.imread(img_path_cam1)
        img_cam3 = cv2.imread(img_path_cam3)

        # CAM1에서 화살 검출 결과가 있으면 우선 사용
        if (
            result_cam1.keypoints is not None
            and result_cam1.keypoints.xy.cpu().numpy().size > 0
        ):
            keypoints = result_cam1.keypoints.xy.cpu().numpy()
            selected_img = img_cam1
            selected_img_name = img_name_cam1
            detected_count = len(keypoints)
            print(f"{selected_img_name}: CAM1에서 객체 검출됨")
        else:
            keypoints = None
            selected_img = img_cam1  # 기본적으로 CAM1 이미지 사용
            selected_img_name = img_name_cam1
            detected_count = 0
            print(f"{selected_img_name}: CAM1, CAM3 모두에서 객체 검출 안됨")

        # 파일 이름에서 "_frame_숫자" 제거하여 기준 이름 추출
        base_name = re.sub(
            r"_frame_\d+\.(png|jpg)$", "", os.path.basename(img_path_cam1)
        )
        # 원근 변환 적용 및 화살 좌표 변환
        transformed_img, transformed_keypoints, _ = (
            transformer.apply_perspective_transform(selected_img, keypoints, base_name)
        )
        if transformed_keypoints is not None:
            latest_keypoint = transformer.index_keypoints(transformed_keypoints)
            x, y = latest_keypoint
        else:
            x, y = 0, 0

        # 변환된 이미지 저장
        transformed_img_path = transformer.save_transformed_image(
            transformed_img, selected_img_name
        )
        # print(f"변환된 이미지 경로: {transformed_img_path}")

        # TargetScorer로 점수 산출
        center, contours = target_detector.get_contours(transformed_img_path)
        if contours != None:  # 컨투어를 10개 찾았으면 추가
            print("$$$새로운 컨투어 사용$$$")
            score = scorer.get_score(center, [(x, y)], contours)
            for c in contours:
                contours_list.append(c)
        else:
            print("@@@백업 컨투어 사용@@@")
            score = scorer.get_score(center, [(x, y)], contours_list)
        # print(f"{selected_img_name}: 점수 = {score}")
        hits.append(
            {
                "point": (
                    int(x),
                    int(y),
                ),
                "score": score,
            }
        )
        # Example usage:
        img = cv2.imread(transformed_img_path)

        # Create the visualizer
        C_x, C_y = center
        visualizer = TargetVisualizer(int(C_x), int(C_y))

        # Draw the visualization
        output_img = visualizer.visualize(img, hits)

        # Display the image
        cv2.imshow("Visualization", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # AccuracyValidator로 검출 결과 업데이트 및 JSON 저장
        accumulated_score = score  # 여기서는 단순히 해당 프레임의 점수를 사용 (누적 방식은 필요에 따라 수정)
        truth_score = validator.update_metrics(
            selected_img_name, detected_count, accumulated_score
        )
        validator.save_results(
            selected_img_name, truth_score, detected_count, accumulated_score
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/pose_s_add_best.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="./testset/20250116_091103/cam1_4set/",
    )
    parser.add_argument(
        "--source1",
        type=str,
        default="./testset/20250116_091103/cam2_4set/",
    )
    parser.add_argument(
        "--perspective", type=str, default="./perspective_coordinates/20250116.txt"
    )
    parser.add_argument("--output", type=str, default="./output_results")
    args = parser.parse_args()
    main(args)
