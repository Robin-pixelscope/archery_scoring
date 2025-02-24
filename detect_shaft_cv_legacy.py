import cv2
import numpy as np


class DETECT_SHAFT:
    def __init__(self, bg_image_path, frame_image_path, cv_params):
        self.bg_image_path = bg_image_path
        self.frame_image_path = frame_image_path
        self.cv_params = cv_params

    def extend_line(self, x1, y1, x2, y2, width, height):
        """
        Extending the line from shaft detection
        """
        # 기울기 계산
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            # x축 경계 연장
            x_start = 0
            y_start = int(m * x_start + b)
            x_end = width
            y_end = int(m * x_end + b)

            # y축 경계 확인
            if y_start < 0 or y_start > height:
                y_start = 0 if y_start < 0 else height
                x_start = int((y_start - b) / m)

            if y_end < 0 or y_end > height:
                y_end = 0 if y_end < 0 else height
                x_end = int((y_end - b) / m)
        else:
            # 수직선 처리
            x_start = x_end = x1
            y_start = 0
            y_end = height

        return x_start, y_start, x_end, y_end

    def shaft_detect(self):
        """
        detecting a arrow shaft from consecutive frames
        """
        bg = cv2.imread(self.bg_image_path, cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(self.frame_image_path, cv2.IMREAD_GRAYSCALE)
        result = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        image_width, image_height = result.shape[1], result.shape[0]

        # 1. 차영상 계산
        diff = cv2.absdiff(bg, frame)
        # cv2.imshow("diff", diff)

        # 2. 이진화 처리
        _, thresh = cv2.threshold(
            diff, self.cv_params.thres_value, 255, cv2.THRESH_BINARY
        )

        # 3. 모폴로지 연산으로 노이즈 제거
        kernel = np.ones(self.cv_params.kernel_size, np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("cleaned", cleaned)

        gausianblur = cv2.GaussianBlur(
            cleaned, self.cv_params.gaussianblur_kernel_size, sigmaX=0, sigmaY=0
        )
        # cv2.imshow("gausianblur", gausianblur)
        sigma_values = self.cv_params.bilateral_sigma_value
        bilateral = cv2.bilateralFilter(
            gausianblur, sigma_values[0], sigma_values[1], sigma_values[2]
        )
        # cv2.imshow("bilateraldiff", bilateral)

        # 4. 연결된 영역 분석 (Connected Components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bilateral, connectivity=8
        )

        # 5. 영역 크기 기반 필터링 (큰 영역만 유지)
        shaft_min_area = self.cv_params.shaft_min_area  # 영역 최소 크기
        filtered = np.zeros_like(labels, dtype=np.uint8)
        for i in range(1, num_labels):  # 0번은 배경
            if stats[i, cv2.CC_STAT_AREA] >= shaft_min_area:
                filtered[labels == i] = 255
        # cv2.imshow("binary", filtered)

        # 6. 엣지 검출 (Canny Edge Detection)
        edges = cv2.Canny(
            filtered, self.cv_params.canny_thres[0], self.cv_params.canny_thres[1]
        )
        # cv2.imshow("edges", edges)

        # 7. 허프 변환을 통한 직선 검출
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.cv_params.houghline_thres,
            minLineLength=self.cv_params.houghline_minLineLength,
            maxLineGap=self.cv_params.houghline_maxLineGap,
        )

        # 8. 결과 이미지 생성
        result = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            y_coords = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = abs((y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float("inf"))
                if slope >= 0.5 or slope < 5:  # 너무 수평하거나 수직에 가까운 직선 제거
                    cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # print(line)
                    if x1 != x2 or y1 != y2:
                        x_start, y_start, x_end, y_end = self.extend_line(
                            x1, y1, x2, y2, image_width, image_height
                        )
                    # cv2.line(result, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    if y1 > y2:
                        y_coords.append([x1, y1])
                        # print(x1, y1)
                        cv2.circle(result, (x1, y1), 5, (0, 0, 255), -1)
                    else:
                        y_coords.append([x2, y2])
                        # print(x2, y2)
                        cv2.circle(result, (x2, y2), 5, (0, 0, 255), -1)
        # print(len(lines))
        bottom_point = self.get_midpoint_from_points(y_coords)
        # bottom_point = max(y_coords, key=lambda point: point[1])
        # print(len(lines))

        # 10. 결과 출력
        cv2.imshow("Filtered Image", filtered)
        cv2.circle(result, (bottom_point[0], bottom_point[1]), 5, (255, 0, 0), -1)
        cv2.imshow("Final Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return bottom_point[0], bottom_point[1]

    def get_midpoint_from_points(self, points):
        """
        y값 기준으로 내림차순 정렬된 점들 중, 조건을 만족하는 두 점의 중점을 반환합니다.

        Args:
            points: (x, y) 형태의 좌표 리스트

        Returns:
            조건을 만족하는 두 점의 중점 (x, y) 튜플, 만족하는 두 점이 없을 경우 None
        """
        # y값 기준 내림차순으로 정렬
        sorted_points = sorted(points, key=lambda point: point[1], reverse=True)

        # 연속된 두 점을 비교하여 거리 조건 만족 시 중점을 반환
        for i in range(len(sorted_points) - 1):
            point1 = sorted_points[i]
            point2 = sorted_points[i + 1]
            x1, y1 = point1
            x2, y2 = point2
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if distance <= self.cv_params.shaft_line_maxdistance:
                # 중점 계산
                midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
                return midpoint  # 조건을 만족하는 첫 중점 반환

        return sorted_points[0]  # 조건을 만족하는 두 점이 없을 경우 None 반환

    def main(self):
        x, y = self.shaft_detect()
        # print(f"Shaft_point : [{x}, {y}]")
        return x, y
