import cv2
import numpy as np
from shapely.geometry import Point, LineString, Polygon


class ASSIGN_SCORE:
    def __init__(self):
        pass

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
                print(f"점수: {score}점\n")
                return score
            return None

        # 중심에서의 교차점 거리를 기준으로 오름차순 정렬
        intersections.sort(key=lambda x: x[1])

        # 만약 화살의 거리가 가장 안쪽 경계보다 작으면, 중심을 11점, 첫 경계를 10점으로 보고 선형 보간하여 점수 할당
        if radius <= intersections[0][1]:
            d_outer = intersections[0][1]
            fraction = radius / d_outer  # 0이면 중심, d_outer이면 경계
            score = 11 - fraction  # 선형 보간: 중심에서 11, 경계에서 10
            print(f"점수: {score:.2f}점")
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
        last_score = 0  # 10 - (len(intersections) - 1)
        print(f"점수: {last_score}점")
        return last_score
