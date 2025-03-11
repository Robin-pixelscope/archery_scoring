import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


class DETECT_TARGET:
    def __init__(
        self,
        frame_image_path,
        circularity_threshold,
        min_area,
        max_area,
        min_length,
        max_length,
        center_tolerance,
        max_contours_count,
    ):
        self.image_path = frame_image_path
        self.circularity_threshold = circularity_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_length = min_length
        self.max_length = max_length
        self.center_tolerance = center_tolerance
        self.max_contours_count = max_contours_count

    def get_hsv_color_mask_polygons(self, red_roi, center, offset):
        """
        주어진 ROI(HSV 이미지)에서 색영역 마스크를 적용해 원형 컨투어(폴리곤)를 검출합니다.

        Args:
            red_roi: 색공간(HSV)로 변환된 관심 영역. 노란색과 빨간색까지 지정함.
            center: (cX_0, cY_0) 형태의 원본 이미지 중심 좌표.
            offset: (x, y) 오프셋으로, roi가 원본 이미지 내에서 시작하는 좌표.

        Returns:
            yellow_red_colormask_polygons: 조건(원형성, 최소 면적)을 만족하는 컨투어 리스트.
        """

        c_X, c_Y = center
        x_offset, y_offset = offset
        color_ranges = {
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Red": ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
            # "Blue": ([100, 100, 50], [140, 255, 255]),
            # "Black": ([0, 0, 0], [180, 255, 80]),
        }
        yellow_red_colormask_polygons = []
        min_area = 500

        for color_name, ranges in color_ranges.items():
            if color_name == "Red":
                mask1 = cv2.inRange(red_roi, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(red_roi, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(red_roi, np.array(ranges[0]), np.array(ranges[1]))

            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 마스크 레이블링 -----------------------------------------------------
            # 임계치 적용 (이미 이진 이미지이지만, 일반적인 경우 임계치 처리가 필요함)
            ret, binary_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # 연결 요소 레이블링 (기본은 8-연결성)
            num_labels, labels_im = cv2.connectedComponents(binary_img)
            # print("총 객체 수 (배경 포함):", num_labels)
            # 배경(레이블 0)을 제외한 각 레이블의 픽셀 개수를 계산하여 가장 큰 객체 찾기
            max_size = 0
            largest_label = 0
            for label in range(1, num_labels):  # 0은 배경이므로 1부터 시작
                size = np.sum(labels_im == label)
                if size > max_size:
                    max_size = size
                    largest_label = label
            # 가장 큰 객체만 남기기 위한 마스크 생성 (가장 큰 객체는 255, 나머지는 0)
            largest_component = np.uint8(labels_im == largest_label) * 255
            # 커널 정의: 커널의 크기와 모양에 따라 효과 달라짐 (여기서는 5x5 사각형 커널 사용)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # closing 연산 적용: dilation 후 erosion
            closing_img = cv2.morphologyEx(largest_component, cv2.MORPH_CLOSE, kernel)

            # cv2.imshow("mask", closing_img)
            # cv2.waitKey()

            # 컨투어 검출
            contours, _ = cv2.findContours(
                closing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                # # ROI의 좌표를 원본 이미지 기준으로 보정
                cnt += [x_offset, y_offset]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                distance = np.sqrt((c_X - cX) ** 2 + (c_Y - cY) ** 2)
                circularity = 4 * np.pi * (area / (perimeter**2))
                if circularity > self.circularity_threshold and area > min_area:
                    if distance < self.center_tolerance:
                        # print("area : ", area, "perimeter : ", perimeter)
                        yellow_red_colormask_polygons.append(cnt)

        # [print(len(colormask_polygons))]
        return yellow_red_colormask_polygons

    def get_scoring_polygons(self, morphed_edges, original_center, ref_areas):
        """
        엣지 이미지에서 검출한 컨투어 중 원형성, 면적, 그리고 원점(색영역 컨투어 중심)과의 거리를 기준으로
        점수영역에 해당하는 컨투어를 선별합니다.

        Args:
            morphed_edges: 모폴로지 연산을 적용한 엣지 이미지.
            original_center: (cX, cY) 형태의 원본 색영역 중심 좌표.
            ref_areas: 색영역 컨투어들의 면적 리스트(최소/최대 비교에 사용).

        Returns:
            merged_polygons: 조건을 만족하는 컨투어들을 추가 후, 병합(merge_polygons_filter_outer)한 결과.
        """
        cX_0, cY_0 = original_center
        polygons = []

        # 1. skimage의 skeletonize 함수는 0과 1로 이루어진 배열을 사용합니다.
        binary = morphed_edges // 255

        # 2. 씨닝(Thinning) 수행: skeletonize 함수가 이진 이미지의 뼈대를 추출합니다.
        skeleton = skeletonize(binary)
        skeleton_8u = (skeleton * 255).astype(np.uint8)
        # cv2.imshow("skeleton", skeleton_8u)
        # cv2.waitKey()

        contours, _ = cv2.findContours(
            skeleton_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            # 조건에 맞지 않으면 무시
            if area < self.min_area or area > self.max_area:
                # print("Area 필터링!!", area)
                continue
            if perimeter < self.min_length or perimeter > self.max_length:
                # print("Perimeter 필터링!!", perimeter)
                continue
            # (추가) 모양 필터링: 예시로 boundingRect를 이용해 종횡비(aspect ratio) 검사
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h != 0 else 0
            # print(aspect_ratio)
            # 예: 지나치게 납작하거나 길쭉한 것은 제거
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                # print("Aspect ratio 필터링!!", aspect_ratio)
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            distance = np.sqrt((cX_0 - cX) ** 2 + (cY_0 - cY) ** 2)
            if distance < self.center_tolerance:
                circularity = 4 * np.pi * (area / (perimeter**2))
                if (
                    circularity > self.circularity_threshold
                    and area < min(ref_areas) - 500
                ):
                    # print("10점")
                    polygons.append(cnt)
                elif (
                    circularity > self.circularity_threshold
                    and min(ref_areas) + 3000 < area < max(ref_areas) - 5000
                ):
                    # print("8점")
                    polygons.append(cnt)
                elif (
                    circularity > self.circularity_threshold
                    and area > max(ref_areas) + 10000
                ):
                    # print("6점부터")
                    # print("area : ", area, "perimeter : ", perimeter)
                    polygons.append(cnt)
                else:
                    continue

        return polygons

    def process_target_detection(self):
        """
        기존 함수에서 색영역 마스크를 이용해 점수영역을 찾는 부분과,
        컨투어 조건으로 점수영역을 찾는 부분을 분리하여 처리합니다.

        Returns:
            center: 검출된 과녁 중심 좌표 (예: 색영역 중 가장 큰 컨투어의 중심).
            score: (예시) 계산된 점수 혹은 병합된 컨투어 정보.
            score_coutour_list: 점수별 컨투어 리스트.
        """
        # 원본 이미지 읽기 및 전처리
        frame = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()
        height, width = frame.shape[:2]
        score_coutour_list = []

        # 이미지 중심 계산
        center_x = width // 2
        center_y = height // 2

        # 크롭 영역 계산 (마진 적용)
        red_margin = 500
        x1_r = max(center_x - red_margin, 0)
        y1_r = max(center_y - red_margin, 0)
        x2_r = min(center_x + red_margin, width)
        y2_r = min(center_y + red_margin, height)
        red_roi = hsv[y1_r:y2_r, x1_r:x2_r]

        # hsv색공간에서 색영역 마스크로 노란색과 빨간색 컨투어(폴리곤) 검출
        yellow_red_colormask_polygons = self.get_hsv_color_mask_polygons(
            red_roi, (center_x, center_y), (x1_r, y1_r)
        )
        # for cnred in yellow_red_colormask_polygons:
        #     cv2.polylines(output, [cnred], True, (0, 255, 0), 2)
        # cv2.imshow("output", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if not yellow_red_colormask_polygons:
            return None, None

        # 색영역 컨투어의 면적 계산 및 중심점 산출 (가장 큰 컨투어 사용)
        colormask_areas = [
            cv2.contourArea(poly) for poly in yellow_red_colormask_polygons
        ]
        # 가장 작은 면적과 가장 큰 면적의 인덱스 찾기
        min_idx = np.argmin(colormask_areas)
        max_idx = np.argmax(colormask_areas)
        moments = cv2.moments(yellow_red_colormask_polygons[min_idx])
        if moments["m00"] != 0:
            cX_0 = int(moments["m10"] / moments["m00"])
            cY_0 = int(moments["m01"] / moments["m00"])
        else:
            cX_0, cY_0 = None, None
            print("컬러마스크 컨투어 중심 계산 실패")

        # # [디버깅] 가장 작은 면적과 가장 큰 면적의 폴리곤을 초록색으로 그리기
        # cv2.polylines(
        #     output, [yellow_red_colormask_polygons[min_idx]], True, (0, 255, 0), 2
        # )
        # cv2.polylines(
        #     output, [yellow_red_colormask_polygons[max_idx]], True, (0, 255, 0), 2
        # )
        # cv2.imshow("output", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        score_coutour_list = [
            yellow_red_colormask_polygons[min_idx],
            yellow_red_colormask_polygons[max_idx],
        ]

        # 2. 엣지 기반 컨투어에서 조건에 따른 점수영역 검출
        # 그레이스케일 이미지 및 엣지 검출
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)
        # edges = cv2.Canny(gray_blurred, 200, 80)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # dilated = cv2.dilate(edges, kernel, iterations=1)
        # morphed_edges = cv2.erode(dilated, kernel, iterations=1)

        center = (cX_0, cY_0)
        # 10점 원 마스킹(숫자 노이즈 제거)
        radius = 92
        mask = np.ones_like(gray_image, dtype=np.uint8) * 255
        cv2.circle(mask, center, radius, (0,), thickness=-1)
        filtered_img = cv2.bitwise_and(gray_image, mask)

        # BG_values ---------------------------------------------------------------
        gaussian_blurred = cv2.GaussianBlur(filtered_img, (3, 3), 2)
        bilateral_filtered = cv2.bilateralFilter(
            gaussian_blurred, d=1, sigmaColor=1, sigmaSpace=150
        )
        edges = cv2.Canny(gaussian_blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        morphed_edges = cv2.erode(dilated, kernel, iterations=1)
        # cv2.imshow("morphed_edges", morphed_edges)
        # cv2.waitKey()
        # -------------------------------------------------------------------------

        scoring_polygons = self.get_scoring_polygons(
            morphed_edges, (cX_0, cY_0), colormask_areas
        )
        scoring_polygons_area_list = []
        for poly in scoring_polygons:
            area = cv2.contourArea(poly)
            scoring_polygons_area_list.append((poly, area))
        scoring_polygons_area_list.sort(key=lambda x: x[1])  # x[1]이 면적

        filtered = []
        # 가장 최근에 추가된 폴리곤의 면적 (초기값: None)
        last_selected_area = None

        for cnt, area in scoring_polygons_area_list:
            if last_selected_area is None:
                # 첫 번째 폴리곤은 무조건 선택
                filtered.append(cnt)
                last_selected_area = area
            else:
                # 이전에 선택된 폴리곤과 면적 비교
                if abs(area - last_selected_area) > 100:
                    # 면적 차이가 임계값을 초과 -> 새로운 그룹
                    filtered.append(cnt)
                    last_selected_area = area
                else:
                    # 면적 차이가 임계값 이하 -> 같은 그룹이므로 현재 폴리곤은 스킵
                    # (이미 더 작은 면적을 가진 폴리곤이 selected 되어 있음)
                    continue
        # print(len(filtered))
        score_coutour_list.extend(filtered)
        # print("최종 폴리곤 개수: ", len(score_coutour_list))
        # for polyline in score_coutour_list:
        #     cv2.polylines(output, [polyline], True, (255, 255, 255), 2)
        # cv2.imshow("out", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if len(score_coutour_list) != self.max_contours_count:
            return (cX_0, cY_0), None

        # print(f"점수 원 갯수: {len(score_coutour_list)}")
        # # [디버깅] 및 시각화
        # for i in score_coutour_list:
        #     cv2.polylines(output, [np.int32(i)], True, (255, 0, 0), 2)
        # cv2.imshow("out,", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return (cX_0, cY_0), score_coutour_list
