import os
import threading

import cv2

from camera import Camera
from videotospeech import VideoToSpeech

ESP32_CAM_URL = "http://10.24.245.237/stream"
FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_frame.jpg")

print("======================================")
print("       BLIND ASSIST SYSTEM")
print("======================================")

camera = Camera(ESP32_CAM_URL)
ai = VideoToSpeech()
camera.start_stream()
ai.initialize_tts()

analysis_condition = threading.Condition()
pending_analysis = None
pending_skipped_frames = 0
last_submitted_frame_id = -1
analysis_stopping = False
displayed_frame_id = -1

print("Camera and AI pipeline started.")
print("Press ESC to exit.")
print("======================================")


def remove_frame_file():
    try:
        os.remove(FRAME_PATH)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(f"[PIPELINE] Temporary image cleanup error: {error}", flush=True)


def run_analysis_worker():
    global pending_analysis, pending_skipped_frames

    while True:
        with analysis_condition:
            while pending_analysis is None and not analysis_stopping:
                analysis_condition.wait()
            if analysis_stopping:
                return

            frame, frame_id = pending_analysis
            skipped_frames = pending_skipped_frames
            pending_analysis = None
            pending_skipped_frames = 0

        if skipped_frames:
            print(
                f"[AI] Skipped {skipped_frames} superseded frames; "
                f"analyzing latest frame #{frame_id}.",
                flush=True,
            )

        try:
            print(f"[AI] Analysis started for frame #{frame_id}.", flush=True)
            if not cv2.imwrite(
                FRAME_PATH,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            ):
                print("[AI] Analysis skipped: could not save captured frame.", flush=True)
                continue

            description = ai.analyze_frame(FRAME_PATH)
            if description:
                print(f"[AI] Result: {description}", flush=True)
                ai.speak(description)
            else:
                print(f"[AI] No description produced for frame #{frame_id}.", flush=True)
        except Exception as error:
            print(f"[AI] Analysis failed for frame #{frame_id}: {error}", flush=True)
        finally:
            remove_frame_file()


analysis_thread = threading.Thread(
    target=run_analysis_worker,
    name="image-analysis",
    daemon=True,
)
analysis_thread.start()

try:
    while True:
        frame, frame_id = camera.get_latest_frame(displayed_frame_id)

        if frame is not None and frame_id != displayed_frame_id:
            cv2.imshow("Blind Assist - Video", frame)
            displayed_frame_id = frame_id

        if frame is not None:
            with analysis_condition:
                if frame_id != last_submitted_frame_id:
                    if pending_analysis is not None:
                        pending_skipped_frames += 1
                    pending_analysis = (frame, frame_id)
                    last_submitted_frame_id = frame_id
                    analysis_condition.notify()

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
finally:
    with analysis_condition:
        analysis_stopping = True
        pending_analysis = None
        analysis_condition.notify_all()

    try:
        camera.release()
    except Exception as error:
        print(f"[CAMERA] Shutdown error: {error}", flush=True)

    analysis_thread.join(timeout=2.0)
    ai.close()
    analysis_thread.join(timeout=3.0)
    if analysis_thread.is_alive():
        print("[AI] Analysis worker did not stop before timeout.", flush=True)

    remove_frame_file()
    try:
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"[UI] OpenCV shutdown error: {error}", flush=True)
    print("\nBlind Assist stopped.")
