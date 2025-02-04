import cv2
import numpy as np
from scipy.optimize import minimize_scalar


class DETECT_TARGET_2:
    def __init__(
        self,
        frame_image_path,
        x,
        y,
        min_area,
        max_area,
        center_tolerance,
        max_ellipses,
    ):
        self.image_path = frame_image_path
        self.shaft_x = x
        self.shaft_y = y
        self.min_area = min_area
        self.max_area = max_area
        self.center_tolerance = center_tolerance
        self.max_ellipses = max_ellipses

    def process_target_detection(self):
        """
        Processes the color-based segmentation of the target.

        Returns:
            center: The coordinates of the target center.
            score: The score of the segmentation.
        """
        frame = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()
        image_center = (output.shape[1] // 2, output.shape[0] // 2)
        color_ranges = {
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Red": ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
            "Blue": ([100, 110, 70], [140, 255, 255]),
            "Black": ([0, 0, 0], [180, 255, 80]),
        }
        height, width = frame.shape[:2]
        # 크롭 영역의 좌표 계산 (마진 적용)
        x1 = max(self.shaft_x - 500, 0)  # 왼쪽 경계
        y1 = max(self.shaft_y - 500, 0)  # 위쪽 경계
        x2 = min(self.shaft_x + 500, width)  # 오른쪽 경계
        y2 = min(self.shaft_y + 500, height)  # 아래쪽 경계
        # 이미지 크롭
        roi = hsv[y1:y2, x1:x2]

        ellipses = []
        polygons = []
        for color_name, ranges in color_ranges.items():
            if color_name == "Red":
                # 빨간색은 두 범위를 처리
                mask1 = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(roi, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                # 다른 색상은 단일 범위
                mask = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))

            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow(f"{color_name} Mask", mask)  # for debugging purpose

            # 컨투어 검출
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

            # ------------------------------------------------------------------------
            # 원형성 임계값 및 최소 면적 설정
            circularity_threshold = 0.7
            min_area = 500  # 최소 크기 제한

            # 원형 컨투어만 필터링하여 그리기
            for cnt_0 in contours:
                cnt_0 += [x1, y1]
                area = cv2.contourArea(cnt_0)
                perimeter = cv2.arcLength(cnt_0, True)
                if perimeter == 0:
                    continue  # 분모가 0이 되는 경우 방지
                circularity = 4 * np.pi * (area / (perimeter**2))
                if circularity > circularity_threshold and area > min_area:
                    cv2.polylines(output, [cnt_0], True, (0, 255, 0), 2)
                    polygons.append(cnt_0)
            # ------------------------------------------------------------------------

        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # 블러링으로 노이즈 제거
        gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)
        edges = cv2.Canny(gray_blurred, 5, 10)

        # 모폴로지 연산으로 끊어진 엣지 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        morphed_edges = cv2.erode(dilated, kernel, iterations=1)
        cv2.imshow("Morped_Edges", morphed_edges)  # for debugging purpose

        cv2.imshow("Edges", edges)  # for debugging purpose
        cv2.imshow("gray", gray_blurred)
        # cv2.imshow("out,", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        polygon_area = []
        for i in polygons:
            ar = cv2.contourArea(i)
            polygon_area.append(ar)

        # 컨투어 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 원형성 임계값 및 최소 면적 설정
        circularity_threshold = 0.7
        min_area = 500  # 최소 크기 제한

        # 원형 컨투어만 필터링하여 그리기
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue  # 분모가 0이 되는 경우 방지
            circularity = 4 * np.pi * (area / (perimeter**2))
            if circularity > circularity_threshold and area > min_area:
                if area < min(polygon_area, area):
                    cv2.polylines(output, [cnt], True, (255, 255, 0), 2)
                elif max(polygon_area) > area > min(polygon_area):
                    cv2.polylines(output, [cnt], True, (255, 255, 0), 2)
                elif area > max(polygon_area):
                    cv2.polylines(output, [cnt], True, (255, 255, 0), 2)
        # ------------------------------------------------------------------------

        # # 타원의 총 개수를 제한
        # # ellipses = ellipses[: self.max_ellipses]
        # # print("Total ellipses:", len(ellipses))
        # # 타원 병합
        # merged_ellipses = self.merge_ellipses(
        #     ellipses, radius_threshold=5, distance_threshold=20
        # )

        # smallest_ellipse = self.find_smallest_major_axis(ellipses)
        # # print("Smallest ellipse:", smallest_ellipse)
        # # cv2.ellipse(output, smallest_ellipse, (0, 255, 0), 2)
        # center = (int(smallest_ellipse[0][0]), int(smallest_ellipse[0][1]))
        # cv2.circle(output, center, 5, (0, 0, 255), -1)
        # ten_pt_ellipse = (
        #     (smallest_ellipse[0][0], smallest_ellipse[0][1]),
        #     (smallest_ellipse[1][0] / 2, smallest_ellipse[1][1] / 2),
        #     smallest_ellipse[2],
        # )
        # ellipses_of_points = []
        # # ellipses_of_points.append(ten_pt_ellipse)
        # for i in range(1, 11):
        #     temp_ellipse = (
        #         ten_pt_ellipse[0],
        #         (ten_pt_ellipse[1][0] * i, ten_pt_ellipse[1][1] * i),
        #         ten_pt_ellipse[2],
        #     )
        #     ellipses_of_points.append(temp_ellipse)
        #     # cv2.ellipse(output, temp_ellipse, (0, 255, 0), 2)
        # # cv2.ellipse(output, ten_pt_ellipse, (0, 255, 0), 2)
        # # return center
        # # print(ellipses_of_points)
        # # print(len(ellipses_of_points))
        # shaft_positions = [(self.shaft_x, self.shaft_y)]
        # distance = self.calculate_distances(center, shaft_positions)
        # score = self.assign_score(distance, ellipses_of_points)
        # # cv2.circle(output, center, int(distance), (255, 0, 0), 3)

        cv2.imshow("out,", output)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # return center, score, merged_ellipses

    def assign_score(self, radius, ellipses_of_points):
        """
        점수별 타원의 면적과 화살 좌표의 거리를 반지름으로 한 원의 면적을 비교하여 점수를 할당하는 함수.

        Parameters:
            radius (list of floats): 중심 좌표와 화살 좌표 간의 거리(반지름)
            ellipses_of_points (list of floats): 점수별 타원 정보

        Returns:
            score (list of floats): 최종 점수
        """
        area_of_shaft_circle = np.pi * radius**2
        larger_ellipses = []  # 면적이 더 큰 타원 리스트
        for idx, ellipse in enumerate(ellipses_of_points):
            _, axes, _ = ellipse
            a = axes[0] / 2  # 장축 반지름
            b = axes[1] / 2  # 단축 반지름

            # 타원의 면적 계산
            ellipse_area = np.pi * a * b

            # 타원 면적이 원의 면적보다 큰 경우 추가
            if ellipse_area > area_of_shaft_circle:
                larger_ellipses.append((idx, ellipse, ellipse_area))

        if not larger_ellipses:
            return None  # 원보다 큰 타원이 없을 경우 None 반환

        # 면적이 가장 작은 타원 반환
        smallest_larger_ellipse = min(larger_ellipses, key=lambda x: x[2])
        idx = smallest_larger_ellipse[0]
        if 0 <= idx <= 9:
            score = 10 - idx
            print(f"점수: {score}점")
            return score

    def calculate_distances(self, center, arrow_positions):
        """
        중심 좌표와 화살 좌표 간의 거리를 계산하는 함수.

        Parameters:
            center (tuple): 중심 좌표 (x, y)
            arrow_positions (list of tuples): 검출된 화살 좌표 리스트 [(x1, y1), (x2, y2), ...]

        Returns:
            distances (list of floats): 중심 좌표와 화살 좌표 간의 거리 리스트
        """
        distances = []
        for arrow in arrow_positions:
            distance = np.sqrt(
                (arrow[0] - center[0]) ** 2 + (arrow[1] - center[1]) ** 2
            )
            distances.append(distance)
            print(
                f"중심 ({center[0]}, {center[1]}) -> 화살 ({arrow[0]}, {arrow[1]}): 거리 {distance:.2f}"
            )
        return distance

    def find_smallest_major_axis(self, ellipses):
        """
        가장 작은 장축 길이를 가진 두 개의 타원을 찾는 함수.

        Parameters:
            ellipses (list of tuples): cv2.fitEllipse 결과 리스트 [(center, (major_axis, minor_axis), angle), ...]

        Returns:
            smallest_ellipse (tuple): 가장 작은 장축 길이를 가진 타원
            second_smallest_ellipse (tuple): 두 번째로 작은 장축 길이를 가진 타원
        """
        # 타원들을 장축 길이 기준으로 정렬
        sorted_ellipses = sorted(
            ellipses, key=lambda x: x[1][0]
        )  # x[1][0]이 major axis length

        # if len(sorted_ellipses) < 2:
        #     raise ValueError("타원이 2개 이상 있어야 합니다.")

        # 가장 작은 타원과 두 번째로 작은 타원 선택
        smallest_ellipse = sorted_ellipses[0]
        # second_smallest_ellipse = sorted_ellipses[1]

        return smallest_ellipse

    def merge_ellipses(self, ellipses, radius_threshold=5, distance_threshold=10):
        merged = []
        used = [False] * len(ellipses)

        for i, (center, axes, angle) in enumerate(ellipses):
            x1, y1, w1, h1, angle1 = (
                center[0],
                center[1],
                axes[0] / 2,
                axes[1] / 2,
                angle,
            )
            if used[i]:
                continue
            current_cluster = [(x1, y1, w1, h1, angle1)]
            used[i] = True
            for j, (center, axes, angle) in enumerate(ellipses):
                x2, y2, w2, h2, angle2 = (
                    center[0],
                    center[1],
                    axes[0] / 2,
                    axes[1] / 2,
                    angle,
                )
                if i != j and not used[j]:
                    # 두 타원이 병합 조건을 만족하는지 확인
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    if (
                        abs(max(w1, h1) - max(w2, h2)) <= radius_threshold
                        and distance <= distance_threshold
                    ):
                        current_cluster.append((x2, y2, w2, h2, angle2))
                        used[j] = True

            # 병합된 타원의 중심, 축 길이, 각도 계산
            avg_x = np.mean([x for x, y, w, h, angle in current_cluster])
            avg_y = np.mean([y for x, y, w, h, angle in current_cluster])
            max_width = max([w for x, y, w, h, angle in current_cluster])
            max_height = max([h for x, y, w, h, angle in current_cluster])
            avg_angle = np.mean(
                [angle for x, y, w, h, angle in current_cluster]
            )  # 평균 각도
            merged.append(((avg_x, avg_y), (max_width, max_height), avg_angle))

        return merged

    def process_edge_based_detection(self):
        frame = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        output = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 블러링으로 노이즈 제거
        gray_blurred = cv2.GaussianBlur(frame, (0, 0), 1.5)
        edges = cv2.Canny(gray_blurred, 20, 50)
        cv2.imshow("Edges", edges)  # for debugging purpose

        # # 모폴로지 연산으로 끊어진 엣지 연결
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # dilated = cv2.dilate(edges, kernel, iterations=1)
        # morphed_edges = cv2.erode(dilated, kernel, iterations=1)
        # cv2.imshow("Morped_Edges", morphed_edges)  # for debugging purpose

        # 컨투어 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ------------------------------------------------------------------------
        # 원형성 임계값 및 최소 면적 설정
        circularity_threshold = 0.7
        min_area = 500  # 최소 크기 제한

        # 원형 컨투어만 필터링하여 그리기
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue  # 분모가 0이 되는 경우 방지
            circularity = 4 * np.pi * (area / (perimeter**2))
            if circularity > circularity_threshold:
                cv2.polylines(output, [cnt], True, (0, 255, 0), 2)

        # ------------------------------------------------------------------------
        # # 컨투어를 길이 기준으로 정렬 (내림차순)
        # sorted_contours = sorted(contours, key=lambda x: len(x), reverse=True)
        # top_contours = sorted_contours[:11]
        radius_ranges = (600, 1920)
        image_center = (output.shape[1] // 2, output.shape[0] // 2)
        ellipses = []
        for cnt in contours:

            # 최소 외접원의 중심과 반지름 계산
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            if radius_ranges[0] < radius < radius_ranges[1] and len(cnt) >= 5:
                # cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                ellipse = cv2.fitEllipse(cnt)
                center = (int(ellipse[0][0]), int(ellipse[0][1]))

                # 타원의 너비와 높이가 유효한지 확인
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    # 타원의 중심점이 이미지 중심에서 반지름 100 이내에 있는지 확인
                    distance = np.sqrt(
                        (center[0] - image_center[0]) ** 2
                        + (center[1] - image_center[1]) ** 2
                    )
                    if distance <= 30:
                        ellipse = cv2.fitEllipse(cnt)
                        # major_axis = ellipse[1][0] / 2  # 긴 반지름
                        # minor_axis = ellipse[1][1] / 2  # 짧은 반지름
                        # el_area = np.pi * major_axis * minor_axis
                        ellipses.append(ellipse)
                        # print(el_area)
                        # if 1500000 > el_area:
                        # cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                        # cv2.circle(output, (int(x), int(y)), radius, (0, 255, 0), 2)

        cv2.imshow("output", output)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # 타원 병합
        merged_ellipses = self.merge_ellipses(
            ellipses, radius_threshold=5, distance_threshold=20
        )
        # for x, y, w, h, angle in merged_ellipses:
        #     cv2.ellipse(
        #         output,
        #         (int(x), int(y)),
        #         (int(w), int(h)),
        #         angle,
        #         0,
        #         360,
        #         (0, 255, 0),
        #         2,
        #     )
        return merged_ellipses
