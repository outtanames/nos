import json
import signal
import sys
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from nos.client import Client
from nos.logging import logger
from rich import print
import threading

from nos.externals.streamlink import open_audio_stream as open_stream
from nos.managers import ModelHandle, ModelManager
from nos import hub


def main(
    streams: Dict[str, str],
    interval: int = 10,
    preferred_quality="720p",
    sample_rate: int = 16_000,
    record_interval: int = 10,
    base_record_dir: str = "/transcripts",
):

    n_bytes = interval * sample_rate * 2  # Factor 2 comes from reading the int16 stream as bytes

    # TODO: Need to set up client to work with multiple model handles on the same model
    print("Loading model...")
    model_id = "distil-whisper/distil-medium.en"

    # Create stream with model manager directly:
    manager = ModelManager(policy=ModelManager.EvictionPolicy.FIFO, max_concurrent_models=8)
    spec = hub.load_spec(model_id)

    transcription_workers = {}

    record_interval_delta = timedelta(seconds=record_interval)

    def transcription_worker(stream_name, stream_url):
        print("Creating transcription worker for stream: ", stream_name)
        # Set up transcription handlers and ffmpeg streams inside this thread worker:
        handler = manager.load(spec)
        ffmpeg_process, streamlink_process = open_stream(stream_url, preferred_quality, sample_rate=sample_rate)

        # This only works in the main thread
        """
        def handler(signum, frame):
            ffmpeg_process.kill()
            if streamlink_process:
                streamlink_process.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        """

        start_time = datetime.now()

        # transcripts/$STREAM_NAME/transcript_at_$TIME.jsonl
        transcript_filename = start_time.strftime("%Y-%m-%d_%H:%M:%S") + ".jsonl"
        full_record_dir = Path(base_record_dir) / stream_name
        Path(full_record_dir).mkdir(exist_ok=True, parents=True)
        print("Saving transcript to: ", Path(full_record_dir).absolute())
        transcript_path = Path(full_record_dir) / transcript_filename
        current_transcript_file = open(transcript_path.absolute(), "w")
        print("Created starting transcript file: ", transcript_path.absolute())

        try:
            while ffmpeg_process.poll() is None:

                # Read audio from ffmpeg stream
                in_bytes = ffmpeg_process.stdout.read(n_bytes)
                if not in_bytes:
                    break

                # Create a temporary mp3 file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    tmp_file.write(in_bytes)
                    tmp_file.flush()

                # Transcribe the audio
                audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
                response: List[Dict[str, Any]] = handler.transcribe_raw(raw=audio, chunk_length_s=5)["chunks"]
                if not len(response):
                    continue

                # (st, _), (_, end) = response[0]["timestamp"], response[-1]["timestamp"]
                # print date time full day, month, year and time
                print(f"[green]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/green]")
                print(f"[bold white]{' '.join([s['text'] for s in response])}[/bold white]")

                entry = {datetime.now().strftime("%Y-%m-%d %H:%M:%S"): [s["text"] for s in response]}
                current_transcript_file.write(json.dumps(entry) + "\n")

                current_time = datetime.now()
                if current_time - start_time > record_interval_delta:
                    # close the transcript file, open another at the current stamp:
                    current_transcript_file.close()
                    transcript_filename = "transcript_at_" + start_time.strftime("%Y-%m-%d_%H:%M:%S") + ".jsonl"
                    transcript_path = Path(full_record_dir) / transcript_filename
                    current_transcript_file = open(transcript_path.absolute(), "w")
                    start_time = current_time

            logger.debug("Stream ended")
        finally:
            ffmpeg_process.kill()
            if streamlink_process:
                streamlink_process.kill()


    for stream_name, stream_url in streams.items():
        transcription_workers[stream_name] = threading.Thread(target=transcription_worker, args=(stream_name, stream_url))
        transcription_workers[stream_name].start()


if __name__ == "__main__":
    streams = {
        "nbc": "https://www.youtube.com/watch?v=i-bRE31SGvY",
        "abc": "https://www.youtube.com/watch?v=agQ4ejUZ7gI"
    }
    main(streams, interval=10, preferred_quality="272p", sample_rate=16_000)
