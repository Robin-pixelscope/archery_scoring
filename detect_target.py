import cv2
import numpy as np
from collections import deque
from shapely.geometry import Point, LineString, Polygon


class DETECT_TARGET:
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

    def get_color_mask_polygons(self, roi, center, offset):
        """
        주어진 ROI(HSV 이미지)에서 색영역 마스크를 적용해 원형 컨투어(폴리곤)를 검출합니다.

        Args:
            roi: 색공간(HSV)로 변환된 관심 영역.
            center: (cX_0, cY_0) 형태의 원본 이미지 중심 좌표.
            offset: (x, y) 오프셋으로, roi가 원본 이미지 내에서 시작하는 좌표.

        Returns:
            colormask_polygons: 조건(원형성, 최소 면적)을 만족하는 컨투어 리스트.
        """
        c_X, c_Y = center
        x_offset, y_offset = offset
        color_ranges = {
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Red": ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
            "Blue": ([100, 110, 70], [140, 255, 255]),
            "Black": ([0, 0, 0], [180, 255, 80]),
        }
        colormask_polygons = []
        circularity_threshold = 0.8
        min_area = 500

        for color_name, ranges in color_ranges.items():
            if color_name == "Red":
                mask1 = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(roi, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))

            # 모폴로지 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # cv2.imshow("mask", mask)
            # cv2.waitKey()

            # 컨투어 검출
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                if circularity > circularity_threshold and area > min_area:
                    if distance < 30:
                        colormask_polygons.append(cnt)

        [print(len(colormask_polygons))]
        return colormask_polygons

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
        circularity_threshold = 0.8
        min_area = 2000
        polygons = []

        contours, _ = cv2.findContours(
            morphed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            distance = np.sqrt((cX_0 - cX) ** 2 + (cY_0 - cY) ** 2)
            if distance < 20:
                circularity = 4 * np.pi * (area / (perimeter**2))
                if circularity > circularity_threshold and area > max(ref_areas):
                    polygons.append(cnt)
                    # # 원본 색영역 중심과의 거리 계산
                    # distance = np.sqrt((cX_0 - cX) ** 2 + (cY_0 - cY) ** 2)
                    # if distance <= 50:
                    #     if area < min(ref_areas) - 5000:
                    #         polygons.append(cnt)
                    #     elif min(ref_areas) + 5000 < area < max(ref_areas) - 5000:
                    #         polygons.append(cnt)
                    #     elif area > max(ref_areas) + 5000:
                    #         polygons.append(cnt)

        # 추가로 병합 및 필터링 (merge_polygons_filter_outer 함수가 이미 정의되어 있다고 가정)
        merged_polygons = self.merge_polygons_filter_outer(
            polygons, radius_threshold=80, center_distance_threshold=30
        )
        # merged_polygons = self.filter_duplicates_by_iou(polygons, iou_threshold=0.85)
        # merged_polygons = self.filter_duplicates_by_radius(
        #     polygons, r_threshold=80, center_dist_threshold=30
        # )
        # merged_polygons = self.fill_and_merge_rings(
        #     polygons, ring_count=10, ring_gap=95.0
        # )
        # merged_polygons = self.merge_polygons_filter_outer_1to1(
        #     polygons,
        #     gap_lower=85,
        #     gap_upper=105,
        #     expected_gap=95,
        #     total_rings=10,
        #     num_points=100,
        # )
        return merged_polygons

    def calculate_middle_points_polygons(self, scale, polygons):
        """
        주어진 컨투어 리스트의 중심점을 계산합니다.

        Args:
            polygons: 컨투어 리스트.

        Returns:
            middle_points: 컨투어의 중심점 리스트.
        """
        # 1. (N, 1, 2) -> (N, 2)로 변환
        polygon_float = polygons.reshape(-1, 2).astype(np.float32)  # (1076, 2)

        # 2. 다각형 중심(centroid) 계산
        cx = np.mean(polygon_float[:, 0])
        cy = np.mean(polygon_float[:, 1])

        # 3. 각 점에 대해 스케일 변환 적용 (벡터 연산으로 처리)
        #    (x', y') = (cx, cy) + s * ( (x, y) - (cx, cy) )
        cnt_scaled_float = (polygon_float - [cx, cy]) * scale + [cx, cy]

        # 4. (N, 2) -> (N, 1, 2)로 복원 (정수형 변환 포함)
        polygon = cnt_scaled_float.reshape(-1, 1, 2).astype(np.int32)
        return polygon

    def extend_line(self, center, arrow, factor=1000):
        """
        중심점에서 화살 좌표 방향으로 factor배 만큼 연장한 직선을 반환합니다.
        center와 arrow를 float형 1차원 배열로 변환하여 (x, y) 튜플 형태로 만듭니다.
        """
        center = np.array(center, dtype=float).flatten()
        arrow = np.array(arrow, dtype=float).flatten()
        direction = arrow - center
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("중심점과 화살 좌표가 동일합니다.")
        direction = direction / norm
        far_point = center + direction * factor
        return LineString([tuple(center), tuple(far_point)])

    def get_intersection_point(self, polygon, line, center, arrow):
        """
        주어진 폴리곤의 경계와 line의 교차점 중, 중심점에서 화살 방향과 일치하는
        교차점을 선택하여 반환합니다.
        """
        inter = line.intersection(polygon.boundary)
        if inter.is_empty:
            return None

        if inter.geom_type == "Point":
            points = [inter]
        elif inter.geom_type == "MultiPoint":
            points = list(inter.geoms)
        else:
            points = [geom for geom in inter.geoms if geom.geom_type == "Point"]

        center_np = np.array(center, dtype=float).flatten()
        arrow_np = np.array(arrow, dtype=float).flatten()
        direction = arrow_np - center_np
        direction = direction / np.linalg.norm(direction)

        valid_points = []
        for pt in points:
            vec = np.array([pt.x, pt.y], dtype=float) - center_np
            if np.dot(vec, direction) > 0:
                valid_points.append(pt)
        if not valid_points:
            return None

        arrow_t = np.dot(arrow_np - center_np, direction)
        best_point = None
        best_diff = float("inf")
        for pt in valid_points:
            pt_t = np.dot(np.array([pt.x, pt.y], dtype=float) - center_np, direction)
            diff = abs(arrow_t - pt_t)
            if diff < best_diff:
                best_diff = diff
                best_point = pt
        return best_point

    def assign_score(self, center, arrow, contours_of_points):
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
        center_np = np.array(center, dtype=float).flatten()
        arrow_np = np.array(arrow, dtype=float).flatten()
        radius = np.linalg.norm(arrow_np - center_np)

        # 중심-화살 직선 연장
        line = self.extend_line(center, arrow)

        # 각 컨투어를 shapely Polygon으로 변환하고,
        # 중심-화살 직선과의 교차점을 구하여 중심으로부터의 거리를 계산
        intersections = []
        for idx, contour in enumerate(contours_of_points):
            poly_pts = np.array(contour, dtype=float)
            if poly_pts.ndim == 3:
                poly_pts = poly_pts.reshape(-1, 2)
            polygon = Polygon(poly_pts)
            inter_pt = self.get_intersection_point(polygon, line, center, arrow)
            if inter_pt is not None:
                d = np.linalg.norm(
                    np.array([inter_pt.x, inter_pt.y], dtype=float) - center_np
                )
                intersections.append((idx, d))

        # 교차점을 찾지 못하면 기존 면적 기반 방식으로 fallback 처리
        if not intersections:
            area_of_shaft_circle = np.pi * (radius**2)
            larger_contours = []
            for idx, contour in enumerate(contours_of_points):
                contour_area = cv2.contourArea(contour)
                if contour_area > area_of_shaft_circle:
                    larger_contours.append((idx, contour, contour_area))
            if not larger_contours:
                return None
            smallest_larger_contour = min(larger_contours, key=lambda x: x[2])
            idx = smallest_larger_contour[0]
            if 0 <= idx <= 9:
                score = 10 - idx
                print(f"점수: {score}점")
                return score
            return None

        # 중심에서의 교차점 거리를 기준으로 오름차순 정렬
        intersections.sort(key=lambda x: x[1])

        # 화살의 거리가 가장 안쪽 경계보다 작으면 최고점 10점 할당
        if radius <= intersections[0][1]:
            score = 10.0
            print(f"점수: {score}점")
            return score

        # 인접한 두 경계 사이에 화살이 위치한 경우 선형 보간으로 점수 산출
        for i in range(len(intersections) - 1):
            d_inner = intersections[i][1]
            d_outer = intersections[i + 1][1]
            if d_inner < radius <= d_outer:
                s_inner = 10 - i  # 내부 경계에 해당하는 점수 (예: 10점, 9점, …)
                s_outer = 10 - (i + 1)  # 외부 경계에 해당하는 점수
                fraction = (radius - d_inner) / (d_outer - d_inner)
                score = s_inner - fraction * (s_inner - s_outer)
                print(f"점수: {score:.2f}점")
                return score

        # 만약 radius가 가장 바깥쪽 경계보다 멀면 가장 낮은 점수 할당
        last_score = 10 - (len(intersections) - 1)
        print(f"점수: {last_score}점")
        return last_score

    def merge_polygons_filter_outer(
        self, polygons, radius_threshold=50, center_distance_threshold=30
    ):
        """
        1) 입력된 폴리곤(혹은 원 파라미터)을 (original_polygon, (cx,cy), r) 형태로 정규화
        2) 각 폴리곤 간 center_distance_threshold, radius_threshold 기준을 만족하면 '인접'으로 간주
        3) BFS/DFS를 이용해 인접한 폴리곤들을 클러스터로 묶음
        4) 각 클러스터 내에서 평균 반지름에서 50 이하 차이나는 폴리곤들 중
        '가장 안쪽(반지름이 작은)' 폴리곤만 최종 결과에 추가
        """
        # ------------------------------------------------------------------
        # 1) 폴리곤 정규화: (original_polygon, (cx, cy), radius) 형태로 변환
        # ------------------------------------------------------------------
        norm_polys = []  # (original, center, radius) 형태로 저장
        for poly in polygons:
            # numpy 배열이면 list로
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()

            # (cx, cy, r) 형태인지, 혹은 컨투어인지 구분
            if isinstance(poly, (list, tuple)):
                if len(poly) == 3 and isinstance(poly[0], (int, float)):
                    cx, cy, r = poly
                    # 원본 컨투어가 없으니 대체 컨투어 생성
                    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
                    original = np.array(
                        [[cx + r * np.cos(t), cy + r * np.sin(t)] for t in theta],
                        dtype=np.int32,
                    )
                else:
                    # 다각형(컨투어) -> minEnclosingCircle 로 (cx,cy,r) 추출
                    cnt = np.array(poly, dtype=np.int32)
                    if cnt.ndim == 3:
                        cnt = cnt.reshape(-1, 2)
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    original = cnt
            else:
                raise ValueError(
                    "Polygon must be tuple/list/ndarray of points or (cx,cy,r)"
                )

            norm_polys.append((original, (cx, cy), r))

        # ------------------------------------------------------------------
        # 2) 각 폴리곤 사이의 '인접성'을 판별해 그래프(인접 리스트) 구성
        # ------------------------------------------------------------------
        n = len(norm_polys)
        adjacency_list = [[] for _ in range(n)]

        for i in range(n):
            _, (cx_i, cy_i), r_i = norm_polys[i]
            for j in range(i + 1, n):
                _, (cx_j, cy_j), r_j = norm_polys[j]
                dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                if (
                    dist <= center_distance_threshold
                    and abs(r_i - r_j) <= radius_threshold
                ):
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)

        # ------------------------------------------------------------------
        # 3) BFS/DFS 로 '연결된(인접한)' 폴리곤들을 하나의 클러스터로 묶기
        # ------------------------------------------------------------------
        visited = [False] * n
        clusters = []
        for i in range(n):
            if not visited[i]:
                queue = deque([i])
                visited[i] = True
                cluster_indices = []
                while queue:
                    idx = queue.popleft()
                    cluster_indices.append(idx)
                    # idx와 인접한 노드 탐색
                    for nbr in adjacency_list[idx]:
                        if not visited[nbr]:
                            visited[nbr] = True
                            queue.append(nbr)
                clusters.append(cluster_indices)

        # ------------------------------------------------------------------
        # 4) 각 클러스터 내에서
        #    - 평균 반지름에서 ±50 이하인 폴리곤들만 후보
        #    - 후보 중 반지름이 가장 작은 폴리곤을 최종 출력 리스트에 추가
        # ------------------------------------------------------------------
        merged_polygons = []
        for cluster in clusters:
            if not cluster:
                continue
            radii = [norm_polys[idx][2] for idx in cluster]  # 반지름 목록
            avg_r = sum(radii) / len(radii)
            # print(f"평균반지름: {avg_r}")

            # 평균 반지름과 50 이하 차이나는 폴리곤들만 후보
            candidates = [
                (idx, norm_polys[idx][2])
                for idx in cluster
                if abs(norm_polys[idx][2] - avg_r) <= 50
            ]
            if candidates:
                # 반지름이 가장 작은(안쪽) 폴리곤 선택
                best_idx, best_r = min(candidates, key=lambda x: x[1])
                best_original, _, _ = norm_polys[best_idx]
                merged_polygons.append(best_original)
            # print(f"후보: {candidates} / 결과: {best_idx}")

        # print(merged_polygons)
        return merged_polygons

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
        margin = 500
        x1 = max(center_x - margin, 0)
        y1 = max(center_y - margin, 0)
        x2 = min(center_x + margin, width)
        y2 = min(center_y + margin, height)
        roi = hsv[y1:y2, x1:x2]

        # 1. 색영역 마스크로 컨투어(폴리곤) 검출
        colormask_polygons = self.get_color_mask_polygons(
            roi, (center_x, center_y), (x1, y1)
        )
        # for ma in colormask_polygons:
        #     cv2.polylines(output, [ma], True, (0, 255, 0), 2)
        # cv2.imshow("output", output)
        # cv2.waitKey()
        if not colormask_polygons:
            return None, None

        # 색영역 컨투어의 면적 계산 및 중심점 산출 (가장 큰 컨투어 사용)
        colormask_areas = [cv2.contourArea(poly) for poly in colormask_polygons]
        # 가장 작은 면적과 가장 큰 면적의 인덱스 찾기
        min_idx = np.argmin(colormask_areas)
        max_idx = np.argmax(colormask_areas)
        moments = cv2.moments(colormask_polygons[min_idx])
        if moments["m00"] != 0:
            cX_0 = int(moments["m10"] / moments["m00"])
            cY_0 = int(moments["m01"] / moments["m00"])
        else:
            cX_0, cY_0 = None, None
            print("컬러마스크 컨투어 중심 계산 실패")

        # [디버깅] 가장 작은 면적과 가장 큰 면적의 폴리곤을 초록색으로 그리기
        cv2.polylines(output, [colormask_polygons[min_idx]], True, (0, 255, 0), 2)
        cv2.polylines(output, [colormask_polygons[max_idx]], True, (0, 255, 0), 2)
        cv2.imshow("output", output)
        cv2.waitKey()

        # approxPolyDP로 Contour 근사 ------------------------------------------------
        # epsilon = arcLength의 일정 비율. 값이 작을수록 원래 윤곽과 가깝게, 클수록 단순화.
        # epsilon = 0.001 * cv2.arcLength(colormask_polygons[min_idx], True)
        # approx = cv2.approxPolyDP(colormask_polygons[min_idx], epsilon, True)
        # cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)  # 근사 컨투어(초록)
        # ---------------------------------------------------------------------------

        # 컬러마스크로 검출한 9점, 7점원을 이용해서 10점원, 8점원, 6점원을 비례계산
        seven_pt = colormask_polygons[max_idx]
        six_pt = self.calculate_middle_points_polygons(1.25, seven_pt)
        eight_pt = self.calculate_middle_points_polygons(0.75, seven_pt)
        nine_pt = colormask_polygons[min_idx]
        ten_pt = self.calculate_middle_points_polygons(0.5, nine_pt)
        score_coutour_list = [six_pt, seven_pt, eight_pt, nine_pt, ten_pt]

        # # [디버깅] 및 시각화
        # cv2.polylines(output, [ten_pt], True, (0, 255, 0), 2)  # 10점원
        # cv2.polylines(output, [eight_pt], True, (0, 255, 0), 2)  # 8점원
        # cv2.polylines(output, [six_pt], True, (0, 255, 0), 2)  # 6점원
        # cv2.imshow("colomask", output)
        # cv2.waitKey()

        # 2. 엣지 기반 컨투어에서 조건에 따른 점수영역 검출
        # 그레이스케일 이미지 및 엣지 검출
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)
        edges = cv2.Canny(gray_blurred, 200, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        morphed_edges = cv2.erode(dilated, kernel, iterations=1)
        # cv2.imshow("morphed_edges", morphed_edges)

        scoring_polygons = self.get_scoring_polygons(
            morphed_edges, (cX_0, cY_0), colormask_areas
        )
        # for i in scoring_polygons:
        #     contour = np.array(i, dtype=np.int32).reshape(-1, 2)
        #     cv2.polylines(output, [contour], True, (255, 0, 0), 2)
        # cv2.imshow("out,", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print(len(scoring_polygons))
        colormask_max_area = cv2.contourArea(seven_pt)  # + 5000
        # 각 폴리곤의 면적을 구하고 (면적, 폴리곤) 튜플로 묶기
        polygons_with_area = []
        for poly in scoring_polygons:
            area = cv2.contourArea(poly)  # OpenCV contour 면적 계산
            if area > colormask_max_area:
                polygons_with_area.append((area, poly))

        # 면적이 작은 순으로 정렬
        polygons_with_area.sort(key=lambda x: x[0])

        # 정렬된 리스트에서 상위 6개만 추출
        top_6 = polygons_with_area[:5]

        # 최종 점수 영역 폴리곤들만 추출 (폴리곤만 꺼내서 새 리스트 구성)
        final_scoring_polygons = [item[1] for item in top_6]
        score_coutour_list.extend(final_scoring_polygons)
        # print(len(final_scoring_polygons[0]))

        shaft_positions = [(self.shaft_x, self.shaft_y)]
        score = self.assign_score((cX_0, cY_0), shaft_positions, score_coutour_list)
        print(f"점수 원 갯수: {len(score_coutour_list)}")
        # [디버깅] 및 시각화
        for i in score_coutour_list:
            cv2.polylines(output, [np.int32(i)], True, (255, 0, 0), 2)
        cv2.imshow("out,", output)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return (cX_0, cY_0), score, score_coutour_list
