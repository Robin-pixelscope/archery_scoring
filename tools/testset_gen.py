import subprocess
import os

# 입력 비디오 파일 및 출력 폴더 설정
input_video = "/home/robin/다운로드/20250122_145719/CAM03.mp4"
output_folder = "/home/robin/다운로드/20250122_145719/cam3"

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# ffmpeg 명령어 실행 (1초에 한 번 프레임 추출)
subprocess.run(
    [
        "ffmpeg",
        "-i",
        input_video,
        "-vf",
        "fps=1/4",  # 초당 1프레임 추출
        os.path.join(output_folder, "frame_%04d.png"),
    ]
)

print(f"Frames saved to '{output_folder}'")
