import cv2


def capture_frames(video_path, output_path, interval_seconds):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_interval = int(fps * interval_seconds)

    print(
        f"Total frames: {total_frames} - FPS: {fps} - Frames per interval: {frames_per_interval}"
    )

    counter = 972  # CAMBIAR

    for frame_num in range(0, total_frames, frames_per_interval):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_num}")
            break

        file_name = str(counter).zfill(5)
        output_file = f"{output_path}{file_name}.jpg"
        cv2.imwrite(output_file, frame)

        print(f"Frame {counter} saved to {output_file}")
        counter += 1

    cap.release()


if __name__ == "__main__":

    video_path = "videos/clip17.mkv"

    output_path = "images/"

    interval_seconds = 10

    capture_frames(video_path, output_path, interval_seconds)
