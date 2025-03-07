import cv2
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict
from detect_shaft_cv_legacy import DETECT_SHAFT
from detect_target import DETECT_TARGET
from detect_target_legacy import DETECT_TARGET_LEGACY
from detect_target_test import DETECT_TARGET_TEST
from visualize import TargetVisualizer

from collections import defaultdict, deque


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
    hits = []

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

    # # LEGACY-----------------------------------------------------------------------------
    # target_detector = DETECT_TARGET_LEGACY(
    #     image_path,
    #     x,
    #     y,
    #     min_area=5000,
    #     max_area=1000000,
    #     center_tolerance=300,
    #     max_ellipses=15,
    # )
    # center, score, color_ellipses = target_detector.process_color_based_segmentation()
    # edge_ellipses = target_detector.process_edge_based_detection()
    # output = cv2.imread(image_path)
    # circle_8 = (
    #     color_ellipses[0][0],
    #     (color_ellipses[0][1][0] * 3, color_ellipses[0][1][1] * 3),
    #     color_ellipses[0][2],
    # )
    # circle_6 = (
    #     color_ellipses[0][0],
    #     (color_ellipses[0][1][0] * 5, color_ellipses[0][1][1] * 5),
    #     color_ellipses[0][2],
    # )
    # cv2.ellipse(output, color_ellipses[0], (0, 255, 0), 2)  # 10점원
    # cv2.ellipse(output, circle_8, (0, 255, 0), 2)  # 8점원
    # cv2.ellipse(output, circle_6, (0, 255, 0), 2)  # 6점원
    # for c_el in color_ellipses:
    #     c_el = (c_el[0], (c_el[1][0] * 2, c_el[1][1] * 2), c_el[2])
    #     cv2.ellipse(output, c_el, (0, 255, 0), 2)
    # for e_el in edge_ellipses:
    #     e_el = (e_el[0], (e_el[1][0] * 2, e_el[1][1] * 2), e_el[2])
    #     cv2.ellipse(output, e_el, (255, 0, 0), 2)
    #     cv2.ellipse(output, e_el, (0, 255, 0), 2)
    # cv2.imshow("output", output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # # -----------------------------------------------------------------------------------

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
    home_dir = "./testset/20250116_091103/cam1_4set_warped"
    # 폴더에서 파일 이름 읽기
    all_files = sorted(os.listdir(home_dir))
    # 처리할 이미지 확장자 목록 (필요에 따라 추가)
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # pair_list = select_input_images(all_files)

    contours_list = deque(maxlen=1)
    for i in all_files:
        if os.path.splitext(i)[1].lower() in valid_extensions:
            image_path = os.path.join(home_dir, i)
            print(image_path)
            # frame_image_path = os.path.join(home_dir, i[1])
            # print(f"Processing: {bg_image_path}, {frame_image_path}")
            (cX_0, cY_0), score, contours_of_points = main(image_path)
            if contours_of_points != None:
                contours_list.append(contours_of_points)
            # print(len(contours_of_points))
            print(contours_list)
            # 작성중########################################################################################
            if score == None:
                x, y = 1054, 1081
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
                score = target_detector.assign_score(
                    (cX_0, cY_0), [(x, y)], contours_of_points
                )
            print(score)
            # output_path = os.path.join(home_dir, "results2_thin+mask/" + i)
            # cv2.imwrite(output_path, out)

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
