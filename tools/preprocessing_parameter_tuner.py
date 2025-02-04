import cv2
import numpy as np


def tune_preprocessing_parameters(bg_image_path, frame_image_path):
    """
    이미지 차영상 후 (가우시안 → 바이레터럴 → Canny 엣지 → 모폴로지) 순차적으로 적용한
    최종 결과를 GUI 트랙바로 파라미터를 조정하며 확인할 수 있는 함수.

    Args:
        bg_image_path (str): 배경 이미지(그레이스케일) 경로
        frame_image_path (str): 현재 프레임 이미지(그레이스케일) 경로
    """

    # 이미지를 그레이스케일로 읽기
    bg = cv2.imread(bg_image_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(frame_image_path, cv2.IMREAD_GRAYSCALE)

    if bg is None or frame is None:
        print("이미지를 불러올 수 없습니다. 경로를 다시 확인하세요.")
        return

    # 차영상
    diff = cv2.absdiff(bg, frame)

    # 트랙바 콜백 (실질 동작 없음)
    def nothing(x):
        pass

    # 윈도우 생성
    cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)

    # 트랙바 생성
    cv2.createTrackbar(
        "Gaussian ksize", "Filtered Image", 1, 50, nothing
    )  # 가우시안 커널 크기
    cv2.createTrackbar(
        "Bilateral d", "Filtered Image", 1, 50, nothing
    )  # 바이레터럴 필터 d
    cv2.createTrackbar("Bilateral sigmaColor", "Filtered Image", 1, 150, nothing)
    cv2.createTrackbar("Bilateral sigmaSpace", "Filtered Image", 1, 150, nothing)
    cv2.createTrackbar(
        "Canny T1", "Filtered Image", 50, 255, nothing
    )  # Canny Threshold1
    cv2.createTrackbar(
        "Canny T2", "Filtered Image", 150, 255, nothing
    )  # Canny Threshold2
    cv2.createTrackbar(
        "Morph ksize", "Filtered Image", 1, 50, nothing
    )  # 모폴로지 커널 크기
    cv2.createTrackbar("Operation (0=Erode, 1=Dilate)", "Filtered Image", 0, 1, nothing)

    while True:
        # 트랙바 값 읽기
        g_ksize = cv2.getTrackbarPos("Gaussian ksize", "Filtered Image") * 2 + 1  # 홀수
        b_d = cv2.getTrackbarPos("Bilateral d", "Filtered Image")
        b_sigmaColor = cv2.getTrackbarPos("Bilateral sigmaColor", "Filtered Image")
        b_sigmaSpace = cv2.getTrackbarPos("Bilateral sigmaSpace", "Filtered Image")
        canny_t1 = cv2.getTrackbarPos("Canny T1", "Filtered Image")
        canny_t2 = cv2.getTrackbarPos("Canny T2", "Filtered Image")
        morph_ksize = cv2.getTrackbarPos("Morph ksize", "Filtered Image") * 2 + 1
        operation = cv2.getTrackbarPos(
            "Operation (0=Erode, 1=Dilate)", "Filtered Image"
        )

        # --- 가우시안 블러 ---
        gaussian_blur = cv2.GaussianBlur(frame, (g_ksize, g_ksize), 0)

        # --- 바이레터럴 필터 ---
        if b_d < 1:
            b_d = 1  # d가 0이면 오류 발생할 수 있으므로 최소 1로 제한
        bilateral_filtered = cv2.bilateralFilter(
            gaussian_blur, b_d, b_sigmaColor, b_sigmaSpace
        )

        # --- Canny 엣지 ---
        edges = cv2.Canny(bilateral_filtered, canny_t1, canny_t2)

        # --- 모폴로지 연산 (침식/팽창) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        if operation == 0:
            # 침식
            morph_result = cv2.erode(edges, kernel, iterations=1)
        else:
            # 팽창
            morph_result = cv2.dilate(edges, kernel, iterations=1)

        # --- 결과 표시 (최종 모폴로지 결과) ---
        cv2.imshow("Filtered Image", morph_result)

        # --- ESC 키로 종료 ---
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 예시 실행
    bg_image_path = "/home/robin/code/archery_scoring/testset/20250122_145719/cam2/warped_frame_0033.png"
    frame_image_path = "/home/robin/code/archery_scoring/testset/20250122_145719/cam2/warped_frame_0038.png"
    tune_preprocessing_parameters(bg_image_path, frame_image_path)
