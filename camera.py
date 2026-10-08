import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np


class Camera:
    READ_TIMEOUT = 3.0
    READ_SIZE = 4096
    MAX_JPEG_SIZE = 8 * 1024 * 1024

    def __init__(self, stream_url):
        self.stream_url = stream_url
        self._frame_lock = threading.Lock()
        self._response_lock = threading.Lock()
        self._active_response = None
        self._latest_frame = None
        self._latest_frame_id = 0
        self._stop_event = threading.Event()
        self._stream_thread = None

        print("[CAMERA] ESP32-CAM HTTP stream mode initialized.", flush=True)

    def start_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            return

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="camera-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def _set_active_response(self, response):
        with self._response_lock:
            self._active_response = response

    def _clear_active_response(self, response):
        with self._response_lock:
            if self._active_response is response:
                self._active_response = None

    def _stream_loop(self):
        reconnect_delay = 0.5

        while not self._stop_event.is_set():
            response = None
            try:
                request = Request(
                    self.stream_url,
                    headers={"User-Agent": "BlindAssist/1.0"},
                )
                response = urlopen(request, timeout=self.READ_TIMEOUT)
                self._set_active_response(response)

                status = response.getcode()
                if status is not None and not 200 <= status < 300:
                    raise OSError(f"HTTP stream returned status {status}")

                print(f"[CAMERA] Connected to stream: {self.stream_url}", flush=True)
                reconnect_delay = 0.5
                buffer = bytearray()
                frame_started_at = None
                last_timing_log = 0.0

                while not self._stop_event.is_set():
                    read_started = time.perf_counter()
                    chunk = response.read1(self.READ_SIZE)
                    read_finished = time.perf_counter()
                    if not chunk:
                        raise ConnectionError("HTTP stream closed by the camera")
                    buffer.extend(chunk)

                    while not self._stop_event.is_set():
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            keep = 1 if buffer.endswith(b"\xff") else 0
                            if keep:
                                buffer[:] = buffer[-1:]
                            else:
                                buffer.clear()
                            frame_started_at = None
                            break
                        if start:
                            del buffer[:start]
                        if frame_started_at is None:
                            frame_started_at = read_started

                        end = buffer.find(b"\xff\xd9", 2)
                        if end < 0:
                            if len(buffer) > self.MAX_JPEG_SIZE:
                                print(
                                    "[CAMERA] Discarding oversized/incomplete JPEG frame.",
                                    flush=True,
                                )
                                del buffer[:2]
                            break

                        jpeg = bytes(buffer[: end + 2])
                        del buffer[: end + 2]
                        received_at = time.perf_counter()
                        frame_receive_time = received_at - frame_started_at
                        frame_started_at = None
                        decode_started = time.perf_counter()
                        try:
                            frame = cv2.imdecode(
                                np.frombuffer(jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                        except cv2.error as error:
                            print(f"[CAMERA] JPEG decode error: {error}", flush=True)
                            continue
                        decoded_at = time.perf_counter()

                        if frame is None or frame.size == 0:
                            print("[CAMERA] Ignoring an invalid JPEG frame.", flush=True)
                            continue

                        with self._frame_lock:
                            self._latest_frame = frame
                            self._latest_frame_id += 1

                        if decoded_at - last_timing_log >= 1.0:
                            print(
                                "[CAMERA] Frame received in "
                                f"{frame_receive_time:.3f} s "
                                f"(last stream read {read_finished - read_started:.3f} s); "
                                f"decoded in {decoded_at - decode_started:.3f} s "
                                f"({len(jpeg)} bytes).",
                                flush=True,
                            )
                            last_timing_log = decoded_at

            except (URLError, TimeoutError, OSError, ConnectionError) as error:
                if not self._stop_event.is_set():
                    print(f"[CAMERA] Stream error: {error}", flush=True)
            except Exception as error:
                if not self._stop_event.is_set():
                    print(f"[CAMERA] Unexpected stream error: {error}", flush=True)
            finally:
                if response is not None:
                    self._clear_active_response(response)
                    try:
                        response.close()
                    except OSError as error:
                        print(f"[CAMERA] Stream close error: {error}", flush=True)

            if not self._stop_event.is_set():
                print(
                    f"[CAMERA] Reconnecting in {reconnect_delay:.1f} s...",
                    flush=True,
                )
                self._stop_event.wait(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 5.0)

    def get_latest_frame(self, last_frame_id=None):
        with self._frame_lock:
            if (
                self._latest_frame is None
                or self._latest_frame_id == last_frame_id
            ):
                return None, self._latest_frame_id
            return self._latest_frame.copy(), self._latest_frame_id

    def release(self):
        self._stop_event.set()
        with self._response_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except OSError as error:
                print(f"[CAMERA] Stream shutdown error: {error}", flush=True)

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=self.READ_TIMEOUT + 1.0)
            if self._stream_thread.is_alive():
                print("[CAMERA] Stream thread did not stop before timeout.", flush=True)
