import cv2
import numpy as np
import os
import gc


class PerspectiveTransformer:
    def __init__(self, input_folder, output_folder, width=1920, height=1920):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)  # 출력 폴더 생성

        self.image_paths = [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not self.image_paths:
            print("이미지가 폴더에 없습니다.")
            return

        self.points_src = []  # 클릭한 좌표 리스트
        self.first_image_points = None  # 첫 번째 이미지의 좌표 저장
        self.current_image_index = 0  # 현재 처리 중인 이미지 인덱스
        self.resize_ratio_w = 1.0
        self.resize_ratio_h = 1.0
        self.width = width
        self.height = height
        self.temp_image = None

        print(f"처리할 이미지 {len(self.image_paths)}개를 로드했습니다.")
        self.load_image()

    def load_image(self):
        """현재 인덱스의 이미지를 로드하고 축소하여 표시"""
        if self.current_image_index >= len(self.image_paths):
            print("모든 이미지 처리가 완료되었습니다.")
            cv2.destroyAllWindows()
            return

        image = cv2.imread(self.image_paths[self.current_image_index])
        if image is None:
            print(
                f"이미지를 로드하지 못했습니다: {self.image_paths[self.current_image_index]}"
            )
            return

        height, width = image.shape[:2]
        self.resize_ratio_w = 1920 / width if width > 1920 else 1.0
        self.resize_ratio_h = 1080 / height if height > 1080 else 1.0
        resized_image = cv2.resize(
            image, (int(width * self.resize_ratio_w), int(height * self.resize_ratio_h))
        )

        self.temp_image = resized_image.copy()

        cv2.imshow("Select 4 Corners", self.temp_image)

        if self.current_image_index == 0:
            cv2.setMouseCallback("Select 4 Corners", self.mouse_click)
        else:
            self.perform_perspective_transform()  # 첫 번째 이후 이미지는 자동 변환

    def mouse_click(self, event, x, y, flags, param):
        """첫 번째 이미지에서 마우스 클릭으로 좌표 저장"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_image_index == 0:
            original_x = int(x / self.resize_ratio_w)
            original_y = int(y / self.resize_ratio_h)
            self.points_src.append((original_x, original_y))
            print(
                f"클릭한 좌표 (축소): ({x}, {y}) -> 원본 좌표: ({original_x}, {original_y})"
            )

            cv2.circle(self.temp_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Corners", self.temp_image)

            if len(self.points_src) == 4:
                self.first_image_points = (
                    self.points_src.copy()
                )  # 첫 번째 이미지 좌표 저장
                print("첫 번째 이미지 좌표 저장 완료:", self.first_image_points)
                cv2.setMouseCallback(
                    "Select 4 Corners", lambda *args: None
                )  # 콜백 해제
                self.perform_perspective_transform()

    def perform_perspective_transform(self):
        """원근 변환 수행"""
        if self.first_image_points is None:
            print("첫 번째 이미지에서 좌표를 설정해야 합니다.")
            return

        # 정면 뷰 변환 후 원하는 크기
        points_dst = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype="float32",
        )

        # 첫 번째 이미지에서 찍은 좌표를 사용
        src_points = np.array(self.first_image_points, dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_points, points_dst)

        # 원근 변환 적용
        image = cv2.imread(self.image_paths[self.current_image_index])
        if image is None:
            print(
                f"이미지를 로드하지 못했습니다: {self.image_paths[self.current_image_index]}"
            )
            return

        try:
            warped = cv2.warpPerspective(
                image, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR
            )
        except cv2.error as e:
            print(f"cv2.warpPerspective() 실행 중 오류 발생: {e}")
            return

        # 변환된 이미지 저장 (output_folder를 메인에서 설정한 경로 사용)
        output_path = os.path.join(
            self.output_folder,
            f"warped_{os.path.basename(self.image_paths[self.current_image_index])}",
        )
        cv2.imwrite(output_path, warped)
        print(f"원근 변환 완료: {output_path}")

        self.next_image()

    def next_image(self):
        """다음 이미지를 처리"""
        self.current_image_index += 1
        self.load_image()

    def reset_points(self):
        """좌표 초기화"""
        self.points_src.clear()
        self.first_image_points = None
        print("좌표가 초기화되었습니다.")
        self.load_image()

    def release_memory(self):
        """메모리 누수 방지"""
        gc.collect()

    def run(self):
        """프로그램 실행"""
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):  # 좌표 초기화
                self.reset_points()

            elif key == ord("n"):  # 다음 이미지로 이동
                self.next_image()

            elif key == ord("q"):  # 프로그램 종료
                print("프로그램을 종료합니다.")
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    input_folder = "/home/robin/code/archery_scoring/testset/20250116_094453/cam1_4set"
    output_folder = (
        "/home/robin/code/archery_scoring/testset/20250116_094453/cam1_4set_warped/"
    )

    transformer = PerspectiveTransformer(input_folder, output_folder)
    transformer.run()
