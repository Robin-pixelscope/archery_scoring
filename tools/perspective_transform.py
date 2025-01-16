import cv2
import numpy as np
import os
import gc

# 전역 변수 선언
points_src = []  # 현재 이미지 클릭 좌표
first_image_points_src = []  # 첫 번째 이미지 클릭 좌표 저장
current_image_index = 0  # 현재 처리 중인 이미지 인덱스
image_paths = []  # 이미지 경로 리스트
resize_ratio_w = 0.5  # 가로 축소 비율
resize_ratio_h = 0.5  # 세로 축소 비율


def mouse_click(event, x, y, flags, param):
    """마우스 클릭 이벤트로 첫 번째 이미지에서만 좌표 저장"""
    global points_src, first_image_points_src, temp_image, resize_ratio_w, resize_ratio_h

    if current_image_index == 0 and event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x / resize_ratio_w)
        original_y = int(y / resize_ratio_h)
        points_src.append((original_x, original_y))
        print(
            f"클릭한 좌표 (축소): ({x}, {y}) -> 원본 좌표: ({original_x}, {original_y})"
        )

        cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Corners", temp_image)

        if len(points_src) == 4:
            first_image_points_src = points_src.copy()  # 첫 번째 이미지 좌표 저장
            print("첫 번째 이미지 좌표 저장 완료:", first_image_points_src)
            cv2.setMouseCallback("Select 4 Corners", lambda *args: None)  # 콜백 해제
            perform_perspective_transform()


def perform_perspective_transform():
    """원근 변환 수행"""
    global points_src, first_image_points_src, current_image_index, image_paths

    # 정면 뷰 변환 후 원하는 크기 (예: 1920x1920)
    width, height = 1920, 1920
    points_dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )

    # 첫 번째 이미지 좌표 사용
    if current_image_index > 0:
        points_src = first_image_points_src

    src_points = np.array(points_src, dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_points, points_dst)

    # 원근 변환 적용
    print(f"현재 이미지 경로: {image_paths[current_image_index]}")
    image = cv2.imread(image_paths[current_image_index])
    if image is None:
        print(f"이미지를 로드하지 못했습니다: {image_paths[current_image_index]}")
        return
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)

    # 결과 저장
    output_path = f"./testset/RA_cam2/processed/warped_{os.path.basename(image_paths[current_image_index])}"
    cv2.imwrite(output_path, warped)
    print(f"원근 변환 완료: {output_path}")

    if current_image_index == 0:
        points_src.clear()

    next_image()


def next_image():
    """다음 이미지를 처리"""
    global current_image_index, image_paths, temp_image, resize_ratio_w, resize_ratio_h

    current_image_index += 1
    if current_image_index < len(image_paths):
        image = cv2.imread(image_paths[current_image_index])

        # 이미지 축소
        height, width = image.shape[:2]
        resize_ratio_w = 1920 / width if width > 1920 else 1.0
        resize_ratio_h = 1080 / height if height > 1080 else 1.0
        resized_image = cv2.resize(
            image, (int(width * resize_ratio_w), int(height * resize_ratio_h))
        )

        temp_image = resized_image.copy()

        cv2.imshow("Select 4 Corners", temp_image)
        print(f"다음 이미지를 처리합니다: {current_image_index + 1}/{len(image_paths)}")

        if current_image_index > 0:
            perform_perspective_transform()  # 다음 이미지에서 자동으로 변환 실행

    else:
        print("모든 이미지 처리가 완료되었습니다.")
        cv2.destroyAllWindows()

    release_memory()


def reset_points():
    """좌표 초기화"""
    global points_src, first_image_points_src, temp_image

    points_src.clear()
    first_image_points_src.clear()
    print("좌표가 초기화되었습니다.")

    temp_image = temp_image.copy()
    cv2.imshow("Select 4 Corners", temp_image)


def release_memory():
    """메모리 누수 방지"""
    gc.collect()


def main():
    """메인 함수"""
    global image_paths, temp_image, image

    input_folder = "./testset/RA_cam2/"
    output_folder = "./testset/RA_cam2/processed/"
    os.makedirs(output_folder, exist_ok=True)

    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_paths:
        print("이미지가 폴더에 없습니다.")
        return

    print(f"처리할 이미지 {len(image_paths)}개를 로드했습니다.")

    image = cv2.imread(image_paths[current_image_index])

    height, width = image.shape[:2]
    resize_ratio_w = 1920 / width if width > 1920 else 1.0
    resize_ratio_h = 1080 / height if height > 1080 else 1.0
    resized_image = cv2.resize(
        image, (int(width * resize_ratio_w), int(height * resize_ratio_h))
    )

    temp_image = resized_image.copy()

    cv2.imshow("Select 4 Corners", temp_image)
    cv2.setMouseCallback("Select 4 Corners", mouse_click)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            reset_points()

        elif key == ord("n"):
            next_image()

        elif key == ord("q"):
            print("프로그램을 종료합니다.")
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# 1032, 246
# 2841, 289
# 2993, 1920
# 760, 1901

# # 외곽 자동검출후 변환(미완성)-----------------------------------------------------
# import cv2
# import numpy as np


# def detect_corners(image):
#     # 1. 이미지를 Grayscale로 변환
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 2. 블러링으로 노이즈 제거
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 3. Canny 에지 검출
#     edges = cv2.Canny(blurred, 50, 150)
#     # resized_edges = cv2.resize(edges, (1920, 1080))
#     # cv2.imshow("Edges", resized_edges)
#     # cv2.waitKey(0)

#     # 4. 외곽선 검출
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 5. 가장 큰 외곽선 추출 (표적지일 가능성이 높은 영역)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     largest_contour = contours[0]

#     # 6. 외곽선을 근사화하여 꼭지점 검출 (4개의 꼭지점)
#     epsilon = 0.02 * cv2.arcLength(largest_contour, True)
#     approx = cv2.approxPolyDP(largest_contour, epsilon, True)

#     # 7. 사각형 영역인지 확인
#     if len(approx) == 4:
#         return np.array([point[0] for point in approx], dtype="float32")
#     else:
#         raise ValueError("사각형 외곽선을 찾을 수 없습니다.")


# # 1. 이미지 읽기
# image = cv2.imread("./archery/dataset/Data_V1/images/20240912152704_A01_0.jpg")
# output = image.copy()

# # 2. 코너 자동 검출
# try:
#     points_src = detect_corners(image)
#     print("검출된 코너 좌표:", points_src)
# except ValueError as e:
#     print(e)
#     exit()

# # 3. 정면 변환 후 원하는 크기 지정 (예: 1920x1920)
# width, height = 1920, 1920
# points_dst = np.array(
#     [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
# )

# # 4. 원근 변환 행렬 계산
# matrix = cv2.getPerspectiveTransform(points_src, points_dst)

# # 5. 원근 변환 적용
# warped = cv2.warpPerspective(image, matrix, (width, height))

# # 6. 결과 저장
# cv2.imwrite("./archery/dataset/20240912152704_A01_0.jpg", warped)

# # 7. 결과 확인
# cv2.imshow("Original Image", image)
# cv2.imshow("Warped Image", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
