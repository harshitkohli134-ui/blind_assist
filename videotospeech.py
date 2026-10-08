import queue
import threading
import time

import ollama

MODEL = "qwen3-vl:4b-instruct"
_TTS_STOP = object()


class VideoToSpeech:
    def __init__(self):
        self.client = ollama.Client(timeout=180)
        self.tts_engine = None
        self._tts_queue = queue.Queue(maxsize=1)
        self._tts_lock = threading.Lock()
        self._tts_enqueue_lock = threading.Lock()
        self._tts_ready = threading.Event()
        self._tts_thread = None
        self._tts_available = False
        self._tts_closed = False
        self._closed = False
        print(f"[AI] Ollama client initialized for {MODEL}.", flush=True)

    def initialize_tts(self):
        with self._tts_lock:
            if self._tts_closed:
                print("[TTS] Cannot initialize after shutdown.", flush=True)
                return
            if self._tts_thread and self._tts_thread.is_alive():
                return

            self._tts_ready.clear()
            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                name="speech-worker",
                daemon=True,
            )
            self._tts_thread.start()
            print("[TTS] Speech engine initialization started.", flush=True)

    def _tts_worker(self):
        try:
            import pyttsx3

            self.tts_engine = pyttsx3.init(driverName="sapi5")
            self.tts_engine.setProperty("rate", 220)
            self.tts_engine.setProperty("volume", 1.0)
            self._tts_available = True
            print("[TTS] Speech engine initialized.", flush=True)
        except Exception as error:
            print(f"[TTS] Initialization error: {error}", flush=True)
        finally:
            self._tts_ready.set()

        if not self._tts_available:
            while True:
                try:
                    queued_text = self._tts_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if queued_text is not _TTS_STOP:
                        print(
                            "[TTS] Speech skipped: the speech engine is unavailable.",
                            flush=True,
                        )
                finally:
                    self._tts_queue.task_done()
            return

        while True:
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                with self._tts_lock:
                    if self._tts_closed:
                        break
                continue

            try:
                if text is _TTS_STOP:
                    break

                started = time.perf_counter()
                print("[TTS] Speech started", flush=True)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                elapsed = time.perf_counter() - started
                print(f"[TTS] Speech finished in {elapsed:.2f} s", flush=True)
            except Exception as error:
                elapsed = time.perf_counter() - started
                print(f"[TTS] Speech error: {error}", flush=True)
                print(f"[TTS] Speech failed after {elapsed:.2f} s", flush=True)
            finally:
                self._tts_queue.task_done()

        try:
            self.tts_engine.stop()
        except Exception as error:
            print(f"[TTS] Speech engine shutdown error: {error}", flush=True)

    def analyze_frame(self, image_path):
        started = time.perf_counter()
        print("[AI] Ollama inference started", flush=True)
        try:
            response = self.client.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Describe this scene for a blind person in one short sentence. "
                            "Mention only the most important objects, people, obstacles, and their positions. "
                            "Keep it to 8 to 16 words and do not use list formatting."
                        ),
                        "images": [image_path],
                    }
                ],
                think=False,
                options={
                    "temperature": 0.1,
                    "num_predict": 128,
                },
            )
            message = response.message
            description = (getattr(message, "content", "") or "").strip()
            elapsed = time.perf_counter() - started
            print(f"[AI] Ollama inference finished in {elapsed:.2f} s", flush=True)
            if not description:
                thinking = getattr(message, "thinking", "") or ""
                print(
                    "[AI] Ollama returned an empty description "
                    f"(thinking_chars={len(thinking)}, "
                    f"done_reason={getattr(response, 'done_reason', None)!r}, "
                    f"eval_count={getattr(response, 'eval_count', None)}).",
                    flush=True,
                )
                return ""
            return description
        except Exception as error:
            elapsed = time.perf_counter() - started
            print(f"[AI] Ollama error: {error}", flush=True)
            print(f"[AI] Inference failed after {elapsed:.2f} s", flush=True)
            return ""

    def speak(self, text):
        if not text:
            return

        with self._tts_enqueue_lock:
            with self._tts_lock:
                if self._tts_closed:
                    print("[TTS] Speech skipped: speech worker is shutting down.", flush=True)
                    return

            if self._tts_ready.is_set() and not self._tts_available:
                print("[TTS] Speech skipped: the speech engine is unavailable.", flush=True)
                return

            try:
                self._tts_queue.put_nowait(text)
            except queue.Full:
                try:
                    self._tts_queue.get_nowait()
                    self._tts_queue.task_done()
                    print("[TTS] Replaced an outdated queued description.", flush=True)
                except queue.Empty:
                    pass
                try:
                    self._tts_queue.put_nowait(text)
                except queue.Full:
                    print("[TTS] Speech skipped: queue is full.", flush=True)
                    return

            print(f"[TTS] Description queued ({len(text)} characters).", flush=True)

    def close(self):
        with self._tts_lock:
            if self._closed:
                return
            self._closed = True
            self._tts_closed = True
            tts_thread = self._tts_thread

        try:
            self.client._client.close()
        except Exception as error:
            print(f"[AI] Client shutdown error: {error}", flush=True)

        if tts_thread and tts_thread.is_alive():
            with self._tts_enqueue_lock:
                try:
                    self._tts_queue.get_nowait()
                    self._tts_queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._tts_queue.put_nowait(_TTS_STOP)
                except queue.Full:
                    print("[TTS] Could not enqueue speech shutdown signal.", flush=True)

            tts_thread.join(timeout=5.0)
            if tts_thread.is_alive():
                print("[TTS] Speech worker did not stop before timeout.", flush=True)
