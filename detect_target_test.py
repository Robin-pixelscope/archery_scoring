import cv2
import numpy as np
from scipy.optimize import minimize_scalar


class DETECT_TARGET_TEST:
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

    def get_color_mask_polygons(self, roi, offset):
        """
        주어진 ROI(HSV 이미지)에서 색영역 마스크를 적용해 원형 컨투어(폴리곤)를 검출합니다.

        Args:
            roi: 색공간(HSV)로 변환된 관심 영역.
            offset: (x, y) 오프셋으로, roi가 원본 이미지 내에서 시작하는 좌표.

        Returns:
            colormask_polygons: 조건(원형성, 최소 면적)을 만족하는 컨투어 리스트.
        """
        x_offset, y_offset = offset
        color_ranges = {
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Red": ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
            "Blue": ([100, 110, 70], [140, 255, 255]),
            "Black": ([0, 0, 0], [180, 255, 80]),
        }
        colormask_polygons = []
        circularity_threshold = 0.7
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

            # 컨투어 검출
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # ROI의 좌표를 원본 이미지 기준으로 보정
                cnt += [x_offset, y_offset]
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter**2))
                if circularity > circularity_threshold and area > min_area:
                    colormask_polygons.append(cnt)

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
        circularity_threshold = 0.7
        min_area = 500
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
            circularity = 4 * np.pi * (area / (perimeter**2))
            if circularity > circularity_threshold and area > min_area:
                # 원본 색영역 중심과의 거리 계산
                distance = np.sqrt((cX_0 - cX) ** 2 + (cY_0 - cY) ** 2)
                if distance <= 50:
                    if area < min(ref_areas) - 5000:
                        polygons.append(cnt)
                    elif min(ref_areas) + 5000 < area < max(ref_areas) - 5000:
                        polygons.append(cnt)
                    elif area > max(ref_areas) + 5000:
                        polygons.append(cnt)

        # 추가로 병합 및 필터링 (merge_polygons_filter_outer 함수가 이미 정의되어 있다고 가정)
        merged_polygons = self.merge_polygons_filter_outer(
            polygons, radius_threshold=100, center_distance_threshold=20
        )
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

    def assign_score(self, radius, contours_of_points):
        """
        점수별 컨투어(폴리곤)의 면적과,
        화살 좌표의 거리를 반지름으로 하는 원의 면적을 비교하여 점수를 할당하는 함수.

        Parameters:
            radius (float): 중심 좌표와 화살 좌표 간의 거리(반지름)
            contours_of_points (list of ndarray): 점수별 컨투어(폴리곤) 목록
                                                (예: findContours로 얻은 각 점수 영역)

        Returns:
            score (int or None): 최종 점수 (없으면 None)
        """
        # 화살이 만드는 '원'의 면적 (π * r^2)
        area_of_shaft_circle = np.pi * (radius**2)

        larger_contours = []  # 화살 원보다 면적이 더 큰 컨투어 목록
        for idx, contour in enumerate(contours_of_points):
            contour_area = cv2.contourArea(contour)

            # 컨투어 면적이 화살 원 면적보다 큰 경우만 추가
            if contour_area > area_of_shaft_circle:
                larger_contours.append((idx, contour, contour_area))

        # 화살 원보다 큰 컨투어가 전혀 없으면 점수 계산 불가 → None
        if not larger_contours:
            return None

        # '화살 원 면적보다 크지만, 그 중 가장 작은 면적'의 컨투어 선택
        smallest_larger_contour = min(larger_contours, key=lambda x: x[2])
        idx = smallest_larger_contour[0]

        # 0~9번 범위 안에서만 점수를 10 - idx로 계산 (예시)
        if 0 <= idx <= 9:
            score = 10 - idx
            print(f"점수: {score}점")
            return score

        # 이외에는 None 처리 (필요 시 로직 수정 가능)
        return None

    def merge_polygons_filter_outer(
        self, polygons, radius_threshold=5, center_distance_threshold=10
    ):
        """
        입력된 원 모양의 폴리곤(컨투어)들 중에서,
        radius_threshold와 distance_threshold 기준으로 가까운 폴리곤들을 그룹화한 후,
        각 그룹에서 반지름(최소 외접 원의 반지름)이 가장 작은, 즉 안쪽에 위치한 폴리곤을 그대로 선택합니다.

        매개변수:
          polygons: 원 모양 폴리곤 리스트.
                    각 요소는 다음 중 하나의 형태로 주어질 수 있음:
                      - ((cx, cy), radius)
                      - (cx, cy, radius)
                      - 컨투어(점들의 리스트 또는 numpy 배열)
          radius_threshold: 반지름 차이가 이 값 이하이면 비슷한 크기로 판단
          center_distance_threshold: 중심 간의 거리가 이 값 이하이면 가까운 것으로 판단

        반환값:
          merged_polygons: 필터링된(안쪽) 폴리곤들의 리스트 (원본 폴리곤 그대로)
        """
        merged_polygons = []
        used = [False] * len(polygons)
        norm_polys = []  # 각 폴리곤을 (original_polygon, (cx,cy), radius) 형태로 저장

        for poly in polygons:
            # numpy 배열이면 리스트로 변환
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            if isinstance(poly, (list, tuple)):
                # 만약 (cx, cy, radius) 형태인 경우
                if len(poly) == 3 and isinstance(poly[0], (int, float)):
                    center = (poly[0], poly[1])
                    radius = poly[2]
                    # 원 모양의 다각형이 아니라면, 원 모양의 컨투어를 생성하지 않고
                    # 원래 정보만 저장 (추후에 새로 그리지 않고 필터링할 수 있음)
                    original = None  # 원본 폴리곤이 없으므로 나중에 처리할 수 있음.
                    # 여기서는 원본이 없으면, 원 모양의 다각형(컨투어)로 대체할 수 있습니다.
                    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
                    original = np.array(
                        [
                            [
                                center[0] + radius * np.cos(t),
                                center[1] + radius * np.sin(t),
                            ]
                            for t in theta
                        ],
                        dtype=np.int32,
                    )
                else:
                    # 컨투어(점들의 리스트)로 간주
                    cnt = np.array(poly)
                    # cv2.findContours에서 나온 형식일 수 있으므로 (N,1,2) -> (N,2)
                    if cnt.ndim == 3:
                        cnt = cnt.reshape(-1, 2)
                    (center, radius) = cv2.minEnclosingCircle(cnt)
                    original = np.array(poly, dtype=np.int32)  # 원본 폴리곤 그대로 사용
                norm_polys.append((original, (center[0], center[1]), radius))
            else:
                raise ValueError("Polygon is not a tuple, list, or numpy array")

        # 클러스터링: 각 폴리곤(원으로 표현된 정보)들끼리 가까우면 그룹화
        for i, (orig_i, center_i, radius_i) in enumerate(norm_polys):
            if used[i]:
                continue
            current_cluster = [(i, center_i, radius_i)]
            used[i] = True
            for j, (orig_j, center_j, radius_j) in enumerate(norm_polys):
                if i != j and not used[j]:
                    dist = np.sqrt(
                        (center_i[0] - center_j[0]) ** 2
                        + (center_i[1] - center_j[1]) ** 2
                    )
                    if (
                        abs(radius_i - radius_j) <= radius_threshold
                        and dist <= center_distance_threshold
                    ):
                        current_cluster.append((j, center_j, radius_j))
                        used[j] = True

            # 그룹 내에서 반지름이 가장 작은(안쪽) 폴리곤 선택
            best_index, best_center, best_radius = min(
                current_cluster, key=lambda item: item[2]
            )
            best_original, _, _ = norm_polys[best_index]
            merged_polygons.append(best_original)
        return merged_polygons

    def to_shapely_polygon(self, poly):
        """
        (cx, cy, r) 혹은 컨투어(점 리스트/배열)를
        shapely.geometry.Polygon 객체로 변환하는 함수
        """
        if isinstance(poly, (tuple, list)):
            # (cx, cy, r) 형태
            if len(poly) == 3 and all(isinstance(v, (int, float)) for v in poly):
                cx, cy, r = poly
                # 원을 다각형으로 근사
                theta = np.linspace(0, 2 * np.pi, 72)  # 72각형 근사
                pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in theta]
                return Polygon(pts)
            else:
                # 컨투어(점들의 리스트)
                pts = np.array(poly, dtype=np.float32)
                if pts.ndim == 3:
                    pts = pts.reshape(-1, 2)
                return Polygon(pts)
        elif isinstance(poly, np.ndarray):
            # Numpy array인 경우, 컨투어로 간주
            pts = poly.astype(np.float32)
            if pts.ndim == 3:
                pts = pts.reshape(-1, 2)
            return Polygon(pts)
        else:
            raise ValueError("Unsupported polygon format")

    def iou_polygon(self, polyA, polyB):
        """
        두 shapely Polygon의 IoU(Intersection over Union) 계산
        """
        inter_area = polyA.intersection(polyB).area
        union_area = polyA.union(polyB).area
        if union_area == 0:
            return 0
        return inter_area / union_area

    def filter_duplicates_by_iou(self, polygons, iou_threshold=0.8):
        """
        폴리곤 간 IoU(면적 교집합/합집합) 기반으로 중복 제거.
        - IoU가 iou_threshold 이상이면 같은 폴리곤(중복)으로 간주.

        Parameters:
            polygons (list): 폴리곤 리스트. 각 폴리곤은
                            - (cx, cy, r) 또는
                            - 컨투어(점들의 리스트/배열) 형태
            iou_threshold (float): IoU 임계값 (기본 0.8)

        Returns:
            merged (list): 중복이 제거된 폴리곤 목록
        """
        merged = []
        for poly in polygons:
            poly_shapely = self.to_shapely_polygon(poly)
            is_duplicate = False

            for i, m_poly in enumerate(merged):
                m_shapely = self.to_shapely_polygon(m_poly)
                iou_val = self.iou_polygon(poly_shapely, m_shapely)

                # IoU가 임계값 이상이면 "중복"으로 간주
                if iou_val >= iou_threshold:
                    # 필요하다면, '더 작은 반지름'만 남기는 로직 등 커스터마이징 가능
                    # 이 예시에서는 단순히 먼저 들어온 폴리곤 유지, 새로 들어온 것은 스킵
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(poly)

        return merged

    def extract_center_radius(self, poly):
        """
        폴리곤(컨투어) 혹은 (cx, cy, r) 정보를 입력받아
        (cx, cy, r)를 반환하는 헬퍼 함수.
        """
        # 이미 (cx, cy, r) 형태라면 그대로 반환
        if isinstance(poly, (tuple, list)):
            if len(poly) == 3 and all(isinstance(v, (int, float)) for v in poly):
                return (poly[0], poly[1], poly[2])

        # 그렇지 않다면 컨투어(점들의 리스트/배열)로 가정
        cnt = np.array(poly, dtype=np.float32)
        if cnt.ndim == 3:
            cnt = cnt.reshape(-1, 2)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        return (cx, cy, r)

    def filter_duplicates_by_radius(
        self, polygons, r_threshold=80, center_dist_threshold=30
    ):
        """
        중심 좌표와 반지름을 이용해 중복 폴리곤을 제거하는 함수.
        - 이미 추가된 폴리곤과 (dist < dist_threshold) AND (|r - r'| < r_threshold)이면
        같은 원으로 간주하여 중복 제거.
        - '더 작은 반지름'을 유지하고 싶다면, 이미 등록된 폴리곤과 비교 시
        원하는 로직으로 교체 가능.

        Parameters:
            polygons (list): 폴리곤 리스트. 각 폴리곤은
                            - (cx, cy, r) 또는
                            - 컨투어(점들의 리스트/배열) 형태
            center_dist_threshold (float): 중심 좌표 간 거리 허용치
            r_threshold (float): 반지름 차이 허용치

        Returns:
            merged (list): 중복이 제거된 폴리곤 목록
        """
        merged = []
        for poly in polygons:
            cx, cy, r = self.extract_center_radius(poly)
            is_duplicate = False

            for i, m_poly in enumerate(merged):
                mx, my, mr = self.extract_center_radius(m_poly)

                # 중심 거리
                dist = np.sqrt((cx - mx) ** 2 + (cy - my) ** 2)
                # 반지름 차
                diff_r = abs(r - mr)

                if dist < center_dist_threshold and diff_r < r_threshold:
                    # 여기서 '더 작은 반지름'만 남기고 싶다면 아래와 같이 로직 변경 가능
                    # 예) 현재 poly가 더 작으면 merged[i]를 교체
                    if r < mr:
                        merged[i] = poly
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(poly)

        return merged

    def to_center_radius(self, poly):
        """
        poly가 (cx, cy, r) 형태면 그대로 반환,
        아니면 컨투어(점들의 리스트/배열)로 보고 (cx, cy, r)를 추출.
        """
        if isinstance(poly, (list, tuple)):
            # (cx, cy, r) 형태 확인
            if len(poly) == 3 and all(isinstance(v, (int, float)) for v in poly):
                return (float(poly[0]), float(poly[1]), float(poly[2]))
            else:
                # 컨투어로 간주
                cnt = np.array(poly, dtype=np.float32)
                if cnt.ndim == 3:
                    cnt = cnt.reshape(-1, 2)
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                return (cx, cy, r)
        elif isinstance(poly, np.ndarray):
            cnt = poly.astype(np.float32)
            if cnt.ndim == 3:
                cnt = cnt.reshape(-1, 2)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            return (cx, cy, r)
        else:
            raise ValueError("Polygon format not supported")

    def fill_and_merge_rings(self, polygons, ring_count=10, ring_gap=95.0):
        """
        1) 입력된 원(혹은 컨투어)들을 (cx, cy, r)로 정규화
        2) 모든 원의 중심을 평균 내어 대표 중심으로 사용
        3) 가장 작은 r을 '10점' 반지름으로 간주
        4) ring_gap 간격으로 10점부터 1점까지 (총 10개) 반지름을 계산
        5) 각 점수 반지름에 대해, 기존 폴리곤 중 가장 근접한 것(±some_threshold 이내)을 사용
        - 여러 개가 근접하면 가장 작은 r 선택(=중복 제거)
        - 근접한 게 없으면 새로 생성(누락분 보완)

        Returns:
            result (list): 길이 10의 리스트. (cx, cy, r) 형태로 [10점, 9점, ..., 1점] 순서
        """
        # 1) (cx, cy, r) 정규화
        crs = [self.to_center_radius(poly) for poly in polygons]

        if not crs:
            # 아무것도 없다면, 임의의 기본값으로 생성할 수밖에 없음
            # 예: (0,0) 중심, 10점 반지름=50으로 가정
            result = []
            base_r = 50.0
            for i in range(ring_count):
                r = base_r + i * ring_gap
                result.append((0.0, 0.0, r))
            return result

        # 2) 중심점 평균
        cx_vals = [c[0] for c in crs]
        cy_vals = [c[1] for c in crs]
        avg_cx = float(np.mean(cx_vals))
        avg_cy = float(np.mean(cy_vals))

        # 3) 가장 작은 r 찾기 -> 10점으로 간주
        min_r = min(c[2] for c in crs)

        # 4) ring_gap 간격으로 10개 반지름 계산
        #    10점 = min_r, 9점 = min_r + ring_gap, ... 1점 = min_r + 9*ring_gap
        desired_radii = []
        for i in range(ring_count):
            # i=0 -> 10점, i=1 -> 9점, ..., i=9 -> 1점
            # r_10 + i*ring_gap
            ring_r = min_r + i * ring_gap
            desired_radii.append(ring_r)

        # 5) 각 점수(각 desired_radii)에 대해 가장 가까운 폴리곤 찾기
        #    - 근접 threshold를 ring_gap의 절반 정도로 잡는 등 임의로 설정 가능
        #    - 여러 개가 근접하면 r이 가장 작은 것을 선택
        #    - 근접한 것이 없으면 새로 생성
        result = [None] * ring_count
        threshold = ring_gap * 0.5  # 임의 설정 (필요시 조절)

        used_indices = set()  # 이미 사용한 polygon 인덱스
        for i, target_r in enumerate(desired_radii):
            candidates = []
            for idx, (cx, cy, r) in enumerate(crs):
                if idx in used_indices:
                    continue
                # 반지름 차이
                diff_r = abs(r - target_r)
                if diff_r <= threshold:
                    candidates.append((idx, (cx, cy, r)))

            if candidates:
                # 여러 후보 중 r이 작은 순으로 정렬
                candidates.sort(key=lambda x: x[1][2])
                chosen_idx, chosen_cr = candidates[0]
                used_indices.add(chosen_idx)
                result[i] = chosen_cr  # (cx, cy, r)
            else:
                # 근접한 것이 없으면 새로 생성(누락분 보완)
                # 대표 중심(avg_cx, avg_cy)에 target_r 반지름
                result[i] = (avg_cx, avg_cy, target_r)

        # 결과 리스트는 [10점, 9점, ..., 1점] 순서
        return result

    def circle_to_polygon(self, cx, cy, r, num_points=100):
        """
        (cx, cy) 중심, 반지름 r인 원을 num_points각형으로 근사한 다각형(점 배열)로 반환.
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        pts = np.stack((cx + r * np.cos(angles), cy + r * np.sin(angles)), axis=-1)
        return pts.astype(np.float32)

    def merge_polygons_filter_outer_1to1(
        self,
        polygons,
        gap_lower=85,
        gap_upper=105,
        expected_gap=95,
        total_rings=10,
        num_points=100,
    ):
        """
        [설명]
        - 입력된 폴리곤(또는 (cx, cy, r) 파라미터)을 (original, (cx, cy), r) 형태로 정규화합니다.
        - 정규화 후, 반지름 기준 오름차순으로 정렬한 뒤 가장 작은 원(내부 원, 10점)을 시작으로
        순차적으로 1:1 비교하여, 두 원 간 반지름 차이가 85~105 픽셀 내에 있는 경우만 선택합니다.
        - 만약 선택된 원의 개수가 total_rings(기본 10)개 미만이면, 가장 작은 원의 반지름을 기준으로
        expected_gap(기본 95) 간격으로 누락된 원을 새로 생성하여 보완합니다.
        - 최종 결과는 총 total_rings 개의 원 모양 다각형(각각 (num_points, 2) numpy 배열) 리스트로 반환됩니다.

        Parameters:
        polygons: 입력 폴리곤 리스트 (각 항목은 (cx, cy, r) 또는 컨투어(점들의 리스트/배열))
        gap_lower: 두 원 간 최소 반지름 차 (픽셀)
        gap_upper: 두 원 간 최대 반지름 차 (픽셀)
        expected_gap: 이상적인 반지름 차 (픽셀; 누락 보완 시 사용)
        total_rings: 최종으로 필요한 원의 개수 (예, 10개)
        num_points: 원을 근사할 다각형의 점 개수

        Returns:
        merged_polygons: 총 total_rings 개의 원 모양 다각형 리스트
                        (각각 np.array shape=(num_points, 2))
        """
        # 1. 폴리곤 정규화: (original, (cx, cy), r) 형태로 변환
        norm_polys = []
        for poly in polygons:
            # numpy 배열이면 리스트로 변환
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            # (cx, cy, r) 형태인지, 아니면 컨투어인지 구분
            if isinstance(poly, (list, tuple)):
                if len(poly) == 3 and isinstance(poly[0], (int, float)):
                    cx, cy, r = poly
                    # 원형 다각형(컨투어)이 없으므로 임의로 원을 근사
                    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                    original = np.array(
                        [[cx + r * np.cos(t), cy + r * np.sin(t)] for t in theta],
                        dtype=np.float32,
                    )
                else:
                    # 컨투어로 간주: minEnclosingCircle로 (cx,cy,r) 추출
                    cnt = np.array(poly, dtype=np.int32)
                    if cnt.ndim == 3:
                        cnt = cnt.reshape(-1, 2)
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    original = cnt.astype(np.float32)
            else:
                raise ValueError(
                    "Polygon must be tuple/list/ndarray of points or (cx, cy, r)"
                )
            norm_polys.append((original, (cx, cy), r))

        # 폴리곤이 하나도 없으면 기본값 생성 (중심 (0,0), base_r=50)
        if not norm_polys:
            base_center = (0.0, 0.0)
            base_r = 50.0
            merged_polygons = [
                self.circle_to_polygon(
                    base_center[0],
                    base_center[1],
                    base_r + i * expected_gap,
                    num_points,
                )
                for i in range(total_rings)
            ]
            return merged_polygons

        # 2. 반지름 기준 오름차순 정렬
        norm_polys.sort(key=lambda x: x[2])

        # 3. 가장 작은 원을 시작(10점 원)
        base_original, base_center, base_r = norm_polys[0]
        selected = [
            norm_polys[0]
        ]  # 선택된 원 리스트 (각 항목: (original, (cx, cy), r))
        last_r = base_r

        # 4. 나머지 폴리곤들을 순차적으로 1:1 비교 (반지름 차이가 gap_lower~gap_upper 이내인 경우만 선택)
        for entry in norm_polys[1:]:
            current_r = entry[2]
            gap = current_r - last_r
            if gap_lower <= gap <= gap_upper:
                selected.append(entry)
                last_r = current_r
            if len(selected) >= total_rings:
                break

        # 5. 선택된 원의 개수가 total_rings 개 미만이면, 누락된 원은 기본 간격(expected_gap)을 이용해 생성
        final_selected = selected[
            :total_rings
        ]  # 이미 선택된 것 중 최대 total_rings개 사용
        count = len(final_selected)
        if count < total_rings:
            # (10점 원을 기준으로)
            for i in range(count, total_rings):
                new_r = base_r + i * expected_gap
                final_selected.append((None, base_center, new_r))

        # 6. 각 (cx,cy,r) 정보를 원 모양 다각형으로 변환
        merged_polygons = []
        for entry in final_selected:
            center = entry[1]
            r = entry[2]
            poly = self.circle_to_polygon(center[0], center[1], r, num_points)
            merged_polygons.append(poly)

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

        # 크롭 영역 계산 (마진 적용)
        x1 = max(self.shaft_x - 500, 0)
        y1 = max(self.shaft_y - 500, 0)
        x2 = min(self.shaft_x + 500, width)
        y2 = min(self.shaft_y + 500, height)
        roi = hsv[y1:y2, x1:x2]

        # 1. 색영역 마스크로 컨투어(폴리곤) 검출
        colormask_polygons = self.get_color_mask_polygons(roi, (x1, y1))
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

        # [디버깅] 가장 작은 면적과 가장 큰 면적의 폴리곤을 초록색으로 그리기
        cv2.polylines(output, [colormask_polygons[min_idx]], True, (0, 255, 0), 2)
        cv2.polylines(output, [colormask_polygons[max_idx]], True, (0, 255, 0), 2)

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

        # 2. 엣지 기반 컨투어에서 조건에 따른 점수영역 검출
        # 그레이스케일 이미지 및 엣지 검출
        gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)
        edges = cv2.Canny(gray_blurred, 200, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        morphed_edges = cv2.erode(dilated, kernel, iterations=1)

        scoring_polygons = self.get_scoring_polygons(
            morphed_edges, (cX_0, cY_0), colormask_areas
        )
        colormask_max_area = cv2.contourArea(seven_pt) + 5000
        # 각 폴리곤의 면적을 구하고 (면적, 폴리곤) 튜플로 묶기
        polygons_with_area = []
        for poly in scoring_polygons:
            area = cv2.contourArea(poly)  # OpenCV contour 면적 계산
            if area > colormask_max_area:
                polygons_with_area.append((area, poly))

        # 면적이 작은 순으로 정렬
        polygons_with_area.sort(key=lambda x: x[0])

        # 정렬된 리스트에서 상위 6개만 추출
        top_6 = polygons_with_area[:6]

        # 최종 점수 영역 폴리곤들만 추출 (폴리곤만 꺼내서 새 리스트 구성)
        final_scoring_polygons = [item[1] for item in top_6]
        score_coutour_list.extend(final_scoring_polygons)
        shaft_positions = [(self.shaft_x, self.shaft_y)]
        distance = self.calculate_distances((cX_0, cY_0), shaft_positions)
        score = self.assign_score(distance, score_coutour_list)
        # print(len(scoring_polygons))
        # # [디버깅] 및 시각화
        # for i in score_coutour_list:
        #     cv2.polylines(output, [i], True, (255, 0, 0), 2)
        # cv2.imshow("out,", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return (cX_0, cY_0), score, score_coutour_list

    # --------------------------------------------------------------------------------
    # def process_target_detection(self):
    #     """
    #     Processes the color-based segmentation of the target.

    #     Returns:
    #         center: The coordinates of the target center.
    #         score: The score of the segmentation.
    #     """
    #     frame = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     output = frame.copy()
    #     image_center = (output.shape[1] // 2, output.shape[0] // 2)
    #     color_ranges = {
    #         "Yellow": ([20, 100, 100], [30, 255, 255]),
    #         "Red": ([0, 50, 50], [10, 255, 255], [170, 50, 50], [180, 255, 255]),
    #         "Blue": ([100, 110, 70], [140, 255, 255]),
    #         "Black": ([0, 0, 0], [180, 255, 80]),
    #     }
    #     height, width = frame.shape[:2]
    #     # 크롭 영역의 좌표 계산 (마진 적용)
    #     x1 = max(self.shaft_x - 500, 0)  # 왼쪽 경계
    #     y1 = max(self.shaft_y - 500, 0)  # 위쪽 경계
    #     x2 = min(self.shaft_x + 500, width)  # 오른쪽 경계
    #     y2 = min(self.shaft_y + 500, height)  # 아래쪽 경계
    #     # 이미지 크롭
    #     roi = hsv[y1:y2, x1:x2]

    #     ellipses = []
    #     colormask_polygons = []
    #     for color_name, ranges in color_ranges.items():
    #         if color_name == "Red":
    #             # 빨간색은 두 범위를 처리
    #             mask1 = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))
    #             mask2 = cv2.inRange(roi, np.array(ranges[2]), np.array(ranges[3]))
    #             mask = cv2.bitwise_or(mask1, mask2)
    #         else:
    #             # 다른 색상은 단일 범위
    #             mask = cv2.inRange(roi, np.array(ranges[0]), np.array(ranges[1]))

    #         # 모폴로지 연산으로 노이즈 제거
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #         # cv2.imshow(f"{color_name} Mask", mask)  # for debugging purpose

    #         # 컨투어 검출
    #         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         # cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    #         # 원형성 임계값 및 최소 면적 설정
    #         circularity_threshold = 0.7
    #         min_area = 500  # 최소 크기 제한

    #         # 원형 컨투어만 필터링하여 그리기
    #         for cnt_0 in contours:
    #             cnt_0 += [x1, y1]
    #             area_colormask = cv2.contourArea(cnt_0)
    #             perimeter = cv2.arcLength(cnt_0, True)
    #             if perimeter == 0:
    #                 continue  # 분모가 0이 되는 경우 방지
    #             circularity = 4 * np.pi * (area_colormask / (perimeter**2))
    #             if circularity > circularity_threshold and area_colormask > min_area:
    #                 # cv2.polylines(output, [cnt_0], True, (0, 255, 0), 2)
    #                 colormask_polygons.append(cnt_0)
    #         # ------------------------------------------------------------------------

    #     gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    #     # 블러링으로 노이즈 제거
    #     gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1.5)
    #     edges = cv2.Canny(gray_blurred, 200, 80)

    #     # 모폴로지 연산으로 끊어진 엣지 연결
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    #     # # 블러링으로 노이즈 제거
    #     # gray_blurred = cv2.GaussianBlur(gray_image, (3, 3), 1)
    #     # edges = cv2.Canny(gray_blurred, 200, 80)

    #     # # 모폴로지 연산으로 끊어진 엣지 연결
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     dilated = cv2.dilate(edges, kernel, iterations=1)
    #     morphed_edges = cv2.erode(dilated, kernel, iterations=1)
    #     # cv2.imshow("Morped_Edges", morphed_edges)  # for debugging purpose

    #     # cv2.imshow("Edges", edges)  # for debugging purpose
    #     # cv2.imshow("gray", gray_blurred)
    #     # cv2.imshow("out,", output)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()

    #     # 각 폴리곤의 면적 계산 (cv2.contourArea 사용)
    #     colormask_polygon_area = [cv2.contourArea(poly) for poly in colormask_polygons]

    #     # 가장 작은 면적과 가장 큰 면적의 인덱스 찾기
    #     min_idx = np.argmin(colormask_polygon_area)
    #     max_idx = np.argmax(colormask_polygon_area)

    #     # # 가장 작은 면적과 가장 큰 면적의 폴리곤을 초록색으로 그리기
    #     # cv2.polylines(output, [colormask_polygons[min_idx]], True, (0, 255, 0), 2)
    #     # cv2.polylines(output, [colormask_polygons[max_idx]], True, (0, 255, 0), 2)

    #     # approxPolyDP로 Contour 근사
    #     # epsilon = arcLength의 일정 비율. 값이 작을수록 원래 윤곽과 가깝게, 클수록 단순화.
    #     # epsilon = 0.001 * cv2.arcLength(colormask_polygons[min_idx], True)
    #     # approx = cv2.approxPolyDP(colormask_polygons[min_idx], epsilon, True)
    #     # cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)  # 근사 컨투어(초록)

    #     # 컬러마스크로 검출한 9점, 7점원을 이용해서 10점원, 8점원, 6점원을 비례계산
    #     seven_pt_ellipse = cv2.fitEllipse(colormask_polygons[max_idx])
    #     six_pt_ellipse = (
    #         (seven_pt_ellipse[0][0], seven_pt_ellipse[0][1]),
    #         (seven_pt_ellipse[1][0] * 1.25, seven_pt_ellipse[1][1] * 1.25),
    #         seven_pt_ellipse[2],
    #     )
    #     eight_pt_ellipse = (
    #         (seven_pt_ellipse[0][0], seven_pt_ellipse[0][1]),
    #         (seven_pt_ellipse[1][0] * 0.75, seven_pt_ellipse[1][1] * 0.75),
    #         seven_pt_ellipse[2],
    #     )
    #     nine_pt_ellipse = cv2.fitEllipse(colormask_polygons[min_idx])
    #     ten_pt_ellipse = (
    #         (nine_pt_ellipse[0][0], nine_pt_ellipse[0][1]),
    #         (nine_pt_ellipse[1][0] / 2, nine_pt_ellipse[1][1] / 2),
    #         nine_pt_ellipse[2],
    #     )

    #     # cv2.ellipse(output, ten_pt_ellipse, (0, 255, 0), 2)  # 10점원
    #     # cv2.ellipse(output, eight_pt_ellipse, (0, 255, 0), 2)  # 8점원
    #     # cv2.ellipse(output, six_pt_ellipse, (0, 255, 0), 2)  # 6점원

    #     # 과녁지 중심점 계산
    #     Moment_0riginal = cv2.moments(colormask_polygons[max_idx])
    #     if Moment_0riginal["m00"] != 0:
    #         cX_0riginal = int(Moment_0riginal["m10"] / Moment_0riginal["m00"])
    #         cY_0riginal = int(Moment_0riginal["m01"] / Moment_0riginal["m00"])
    #     #     print(f"Centroid: ({cX_0}, {cY_0})")
    #     # exit()

    #     # 컨투어 검출
    #     contours, _ = cv2.findContours(
    #         morphed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     # 원형성 임계값 및 최소 면적 설정
    #     circularity_threshold = 0.7
    #     min_area = 500  # 최소 크기 제한

    #     polygons = []
    #     # 원형 컨투어만 필터링하여 그리기
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         perimeter = cv2.arcLength(cnt, True)
    #         Moment_contour = cv2.moments(cnt)
    #         if Moment_contour["m00"] != 0:
    #             cX = int(Moment_contour["m10"] / Moment_contour["m00"])
    #             cY = int(Moment_contour["m01"] / Moment_contour["m00"])
    #             # print(f"Centroid: ({cX}, {cY})")
    #         if perimeter == 0:
    #             continue  # 분모가 0이 되는 경우 방지
    #         circularity = 4 * np.pi * (area / (perimeter**2))
    #         if circularity > circularity_threshold and area > min_area:
    #             distance = np.sqrt((cX_0riginal - cX) ** 2 + (cY_0riginal - cY) ** 2)
    #             if distance <= 50:
    #                 if area < min(colormask_polygon_area) - 5000:
    #                     # cv2.polylines(output, [cnt], True, (255, 0, 0), 2)
    #                     polygons.append(cnt)
    #                 elif (
    #                     max(colormask_polygon_area) - 5000
    #                     > area
    #                     > min(colormask_polygon_area) + 5000
    #                 ):
    #                     # cv2.polylines(output, [cnt], True, (255, 0, 0), 2)
    #                     polygons.append(cnt)
    #                 elif area > max(colormask_polygon_area) + 5000:
    #                     # cv2.polylines(output, [cnt], True, (255, 0, 0), 2)
    #                     polygons.append(cnt)
    #                 # else:
    #                 #     print(area)
    #                 #     print("---------------------------------------")

    #     merged_polygons = self.merge_polygons_filter_outer(
    #         polygons, radius_threshold=20, center_distance_threshold=20
    #     )
    #     # for i in merged_polygons:
    #     # cv2.polylines(output, [i], True, (255, 0, 0), 2)

    #     # cv2.imshow("out,", output)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #     # return center, score, merged_ellipses
