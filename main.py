import cv2
import os
import time
import numpy as np
import argparse
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict, deque

# from detect_shaft_cv_legacy import DETECT_SHAFT
# from detect_target_legacy import DETECT_TARGET_LEGACY
# from detect_target_test import DETECT_TARGET_TEST
from scoring_module.detect_shaft_points import ArcheryPoseEstimator
from scoring_module.detect_target import DETECT_TARGET
from scoring_module.visualize import TargetVisualizer
from scoring_module.assign_score import ASSIGN_SCORE


def select_input_images(files):
    """
    파일 이름을 날짜/시간 기준으로 그룹화하고, 그룹 내와 그룹 간 파일 쌍을 처리합니다.

    Args:
        files: 파일 이름 리스트

    Returns:
        None (처리 결과를 출력)
    """
    # 파일 그룹화: 두 번째 부분(날짜/시간)을 기준으로 그룹화
    grouped_files = defaultdict(list)
    for file in files:
        timestamp = file.split("_")[1]  # 파일명에서 두 번째 부분 추출
        grouped_files[timestamp].append(file)

    # 그룹별로 파일 정렬
    for timestamp in grouped_files:
        grouped_files[timestamp].sort()

    # 날짜/시간 그룹별로 연속된 파일 처리
    timestamps = sorted(grouped_files.keys())  # 타임스탬프 정렬
    input_image_pair = []
    for i, timestamp in enumerate(timestamps):
        current_group = grouped_files[timestamp]

        # 현재 그룹 내 연속된 두 파일 처리
        for j in range(len(current_group) - 1):
            file1 = current_group[j]
            file2 = current_group[j + 1]
            input_image_pair.append([file1, file2])
            # print(f"Processing within group: {file1}, {file2}")

    return input_image_pair


@dataclass
class CVParams:
    thres_value: int = 25
    kernel_size: Tuple[int, int] = (5, 5)
    gaussianblur_kernel_size: Tuple[int, int] = (5, 5)
    bilateral_sigma_value: List[int] = (5, 15, 15)
    shaft_min_area: int = 500
    canny_thres: List[int] = (15, 150)
    houghline_thres: int = 30
    houghline_minLineLength: int = 150
    houghline_maxLineGap: int = 15
    shaft_line_maxdistance: int = 50

    # # A02_0- Testing
    # thres_value: int = 25
    # kernel_size: Tuple[int, int] = (3, 3)
    # gaussianblur_kernel_size: Tuple[int, int] = (5, 5)
    # bilateral_sigma_value: List[int] = (5, 15, 15)
    # shaft_min_area: int = 500
    # canny_thres: List[int] = (15, 150)
    # houghline_thres: int = 30
    # houghline_minLineLength: int = 150
    # houghline_maxLineGap: int = 15
    # shaft_line_maxdistance: int = 50


def main(image_path):
    cv_params = CVParams()

    # shaft_detector = DETECT_SHAFT(bg_image_path, frame_image_path, cv_params)
    # x, y = shaft_detector.main()
    # x, y = 938, 905
    x, y = 1054, 1081
    shaft_coords = [[862, 1068], [873, 1055], [870, 1186], [961, 1051], [1054, 1081]]

    # DEBUGGING-------------------------------------------------------------------------
    target_detector = DETECT_TARGET(
        image_path,
        x,
        y,
        min_area=5000,
        max_area=1000000,
        center_tolerance=300,
        max_ellipses=15,
    )
    _ = target_detector.process_target_detection()
    return _

    # # Main ===========================================================================
    # for shaft_coord in shaft_coords:
    #     target_detector = DETECT_TARGET(
    #         image_path,
    #         shaft_coord[0],
    #         shaft_coord[1],
    #         min_area=5000,
    #         max_area=1000000,
    #         center_tolerance=300,
    #         max_ellipses=15,
    #     )
    #     # start_time = time.time()
    #     center, score, contour_list = target_detector.process_target_detection()
    #     hits.append(
    #         {
    #             "point": (
    #                 shaft_coord[0],
    #                 shaft_coord[1],
    #             ),
    #             "score": score,
    #         }
    #     )
    # # Example usage:
    # img = cv2.imread(image_path)

    # # Create the visualizer
    # visualizer = TargetVisualizer(center[0], center[1])

    # # Draw the visualization
    # output_img = visualizer.visualize(img, hits)

    # # Display the image
    # cv2.imshow("Visualization", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # =================================================================================

    # # Testing -----------------------------------------------------------------------
    # target_detector = DETECT_TARGET_TEST(
    #     bg_image_path,
    #     x,
    #     y,
    #     min_area=5000,
    #     max_area=1000000,
    #     center_tolerance=300,
    #     max_ellipses=15,
    # )
    # start_time = time.time()
    # target_detector.process_target_detection()
    # # ----------------------------------------------------------------------------


if __name__ == "__main__":
    # main -----------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./pose_s_add_best.pt")
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
    parser.add_argument("--perspective", type=str, default="./20250116.txt")
    parser.add_argument("--output", type=str, default="./output_results")

    args = parser.parse_args()

    arrow_detector = ArcheryPoseEstimator(
        model_path=args.model,
        source=args.source,
        source1=args.source1,
        perspective_file=args.perspective,
        output_dir=args.output,
    )

    perspective_img, shaft_coords = arrow_detector.process_images()

    home_dir = "./cam1_4set_warped"
    # 폴더에서 파일 이름 읽기
    all_files = sorted(os.listdir(home_dir))
    # 처리할 이미지 확장자 목록 (필요에 따라 추가)
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    hits = []

    scoring = ASSIGN_SCORE()

    for coord in shaft_coords:
        (x, y) = coord
    contours_list = deque(maxlen=10)
    for i in all_files:
        if os.path.splitext(i)[1].lower() in valid_extensions:
            image_path = os.path.join(home_dir, i)
            print(image_path)
            # frame_image_path = os.path.join(home_dir, i[1])
            # print(f"Processing: {bg_image_path}, {frame_image_path}")
            target_detector = DETECT_TARGET(
                image_path,
                circularity_threshold=0.85,  # 원형도 임계값
                min_area=10000,  # 너무 작은 영역은 제거
                max_area=10000000,  # 너무 큰 영역은 제거
                min_length=500.0,  # 윤곽 길이(둘레)가 너무 짧은 것 제거
                max_length=20000.0,  # 너무 긴 것도 제거
                center_tolerance=50,  # 중심 좌표 허용 오차
                max_contours_count=10,  # 최대 컨투어 개수
                # debug = False,  # 디버깅 모드
            )
            (cX_0, cY_0), contours_of_points = (
                target_detector.process_target_detection()
            )
            if contours_of_points != None:  # 컨투어를 10개 찾았으면 추가
                score = scoring.assign_score((cX_0, cY_0), [(x, y)], contours_of_points)
                for c in contours_of_points:
                    contours_list.append(c)
            if contours_of_points == None:
                score = scoring.assign_score((cX_0, cY_0), [(x, y)], contours_list)
            hits.append(
                {
                    "point": (
                        x,
                        y,
                    ),
                    "score": score,
                }
            )
            # Example usage:
            img = cv2.imread(image_path)

            # Create the visualizer
            visualizer = TargetVisualizer(cX_0, cY_0)

            # Draw the visualization
            output_img = visualizer.visualize(img, hits)

            # Display the image
            cv2.imshow("Visualization", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # # =================================================================================
        # # data가 [x, y] 형식인지 검사: 길이가 2이고, 각 요소가 숫자인 경우
        # if (
        #     isinstance(self.shaft_coords, list)
        #     and len(self.shaft_coords) == 2
        #     and all(isinstance(v, (int, float)) for v in self.shaft_coords)
        # ):
        #     score = self.assign_score(
        #         (cX_0, cY_0), self.shaft_coords, score_coutour_list
        #     )
        # # data가 리스트의 리스트([[x1, y1], [x2, y2], ...]) 형식이면 각 좌표를 반복 처리
        # elif (
        #     isinstance(self.shaft_coords, list)
        #     and len(self.shaft_coords) > 0
        #     and isinstance(self.shaft_coords[0], list)
        # ):
        #     for point in self.shaft_coords:
        #         # 각 point가 [x, y] 형식인지 검사 (안전 검사)
        #         if isinstance(point, list) and len(point) == 2:
        #             score = self.assign_score((cX_0, cY_0), point, score_coutour_list)
        #         else:
        #             print("잘못된 좌표 형식:", point)
        # else:
        #     print("알 수 없는 형식입니다.")

    # # BG 폴리곤으로 다음프레임에 그리기 ---------------------------------------------
    # home_dir = "./testset/20250116_091103/cam1_4set_warped/warped_frame_0618.png"
    # base_scoring_polygon = main(home_dir)

    # next_home_dir = "./testset/20250116_091103/cam1_5set_warped/warped_frame_0877.png"
    # im = cv2.imread(next_home_dir)
    # for i in base_scoring_polygon:
    #     cv2.polylines(im, [np.array(i)], True, (0, 255, 0), 2)
    # cv2.imshow("output", im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
