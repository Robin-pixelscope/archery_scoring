import cv2
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict
from detect_shaft import DETECT_SHAFT
from detect_target import DETECT_TARGET
from detect_target_2 import DETECT_TARGET_2
from visualize import TargetVisualizer

from collections import defaultdict


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
    # # A02_0
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

    # A03_0
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


def main(bg_image_path, frame_image_path):
    cv_params = CVParams()

    # shaft_detector = DETECT_SHAFT(bg_image_path, frame_image_path, cv_params)
    # x, y = shaft_detector.main()
    x, y = 938, 905

    # Test -----------------------------------------------------------------------
    target_detector = DETECT_TARGET_2(
        bg_image_path,
        x,
        y,
        min_area=5000,
        max_area=1000000,
        center_tolerance=300,
        max_ellipses=15,
    )
    start_time = time.time()
    target_detector.process_target_detection()
    # ----------------------------------------------------------------------------

    # target_detector = DETECT_TARGET(
    #     bg_image_path,
    #     x,
    #     y,
    #     min_area=5000,
    #     max_area=1000000,
    #     center_tolerance=300,
    #     max_ellipses=15,
    # )
    # start_time = time.time()
    # center, score, color_ellipses = target_detector.process_color_based_segmentation()
    # edge_ellipses = target_detector.process_edge_based_detection()
    # output = cv2.imread(bg_image_path)
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
    #     cv2.ellipse(output, e_el, (0, 255, 0), 2)

    # cv2.imwrite(
    #     f"./testset/20250122_145719/results/cam2/{bg_image_path.split('/')[-1]}", output
    # )
    # # cv2.imshow("output", output)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()

    # end_time = time.time()
    # print(f"Elapsed Time: {end_time - start_time:.2f} sec")

    # --------------------------------------------------------------------------------
    # print(f"Color_based : {ellipse_a} \n")
    # print(f"Edge_based : {ellipse_b}")

    # 가장 가까운 타원 찾기
    # min_index, min_distance = target_detector.closest_ellipse(ellipse_a)
    # print(f"Closest Ellipse Index: {min_index}")
    # print(f"Distance to Closest Ellipse: {min_distance:.2f}")

    # # Example usage:
    # img = cv2.imread(frame_image_path)

    # # Create the visualizer
    # visualizer = TargetVisualizer(center[0], center[1])

    # # Example hit points with scores
    # hits = [{"point": (x, y), "score": score}]

    # # Draw the visualization
    # output_img = visualizer.visualize(img, hits)

    # # Display the image
    # cv2.imshow("Visualization", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":

    # 이미지를 읽어옵니다.
    # bg_image_path = "./testset/RA_cam2/processed/warped_20240912153113_A02_bg.jpg"
    # frame_image_path = "./testset/RA_cam2/processed/warped_20240912153113_A02_0.jpg"

    # bg_image_path = "./testset/RA_cam2/processed/warped_20240912153355_A02_2.jpg"
    # frame_image_path = "./testset/RA_cam2/processed/warped_20240912153355_A02_2.jpg"

    # bg_image_path = "./testset/RA_cam1/processed/warped_20240912153113_A01_1.jpg"
    # frame_image_path = "./testset/RA_cam1/processed/warped_20240912153113_A01_2.jpg"

    # bg_image_path = "./testset/RA_cam3/processed/warped_20240912153113_A03_bg.jpg"
    # frame_image_path = "./testset/RA_cam3/processed/warped_20240912153113_A03_0.jpg"

    # bg_image_path = "./testset/RA_cam3/processed/warped_20240912153113_A03_0.jpg"
    # frame_image_path = "./testset/RA_cam3/processed/warped_20240912153113_A03_1.jpg"

    # bg_image_path = "./testset/RA_cam1/processed/warped_20240912153113_A01_bg.jpg"
    # frame_image_path = "./testset/RA_cam1/processed/warped_20240912153113_A01_0.jpg"

    # main(bg_image_path, frame_image_path)

    home_dir = "./testset/20250122_145719/cam2"
    # 폴더에서 파일 이름 읽기
    all_files = os.listdir(home_dir)

    pair_list = select_input_images(all_files)

    for i in pair_list:
        bg_image_path = os.path.join(home_dir, i[0])
        frame_image_path = os.path.join(home_dir, i[1])
        # print(f"Processing: {bg_image_path}, {frame_image_path}")
        main(bg_image_path, frame_image_path)
