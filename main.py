from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import requests
import os
import uuid
import subprocess
import logging
import threading
from google.cloud import storage
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class RenderJob(BaseModel):
    audio_link: str
    video_links: list[str]
    sentences: list[str]
    callback_webhook: str

    target_duration: int = Field(default=600)
    crossfade_time: int = Field(default=1)
    hertz_freq: float = Field(default=432.0)
    tone_volume: float = Field(default=0.15)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def download_from_gcs(gcs_uri: str, output_path: str, storage_client: storage.Client) -> str:
    logger.info(f"Downloading {gcs_uri}...")
    bucket_name = gcs_uri.split('/')[2]
    blob_name = '/'.join(gcs_uri.split('/')[3:])
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(output_path, timeout=300)
    logger.info(f"Download complete: {output_path}")
    return output_path


def get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def stitch_videos_concat(video_paths: list[str], output_path: str):
    if len(video_paths) == 1:
        subprocess.run(["cp", video_paths[0], output_path], check=True)
        return

    concat_file = output_path.replace('.mp4', '_concat.txt')
    with open(concat_file, 'w') as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy", "-an",
        output_path
    ], check=True)
    logger.info("Stitch complete")


def mix_audio(audio_path: str, mixed_audio_path: str, target_duration: int,
              hertz_freq: float, tone_volume: float, work_dir: str):
    """
    Mix music loop with hz tone using raw PCM as intermediate format.
    PCM (wav) requires no encoding — just raw samples — so intermediate
    steps run at 50-100x speed instead of 0.1x with AAC encoding.
    Only encode to AAC once at the very end.
    """
    music_duration = get_duration(audio_path)
    repeat_count = int(target_duration / music_duration) + 2

    # Step 1 — loop music to PCM wav (no encoding overhead)
    music_concat = f"{work_dir}/music_concat.txt"
    with open(music_concat, 'w') as f:
        for _ in range(repeat_count):
            f.write(f"file '{audio_path}'\n")

    looped_music_wav = f"{work_dir}/music_looped.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", music_concat,
        "-t", str(target_duration),
        "-c:a", "pcm_s16le",   # raw PCM — no encoding, very fast
        looped_music_wav
    ], check=True)
    logger.info("Music loop complete")

    # Step 2 — generate hz tone as PCM wav (sine filter, native C speed)
    tone_wav = f"{work_dir}/tone.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency={hertz_freq}:duration={target_duration}",
        "-af", f"volume={tone_volume}",
        "-c:a", "pcm_s16le",   # raw PCM — no encoding
        tone_wav
    ], check=True)
    logger.info("Tone generation complete")

    # Step 3 — mix and encode to AAC once (only one encode pass total)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", looped_music_wav,
        "-i", tone_wav,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest[out]",
        "-map", "[out]",
        "-c:a", "aac",
        "-b:a", "128k",
        mixed_audio_path
    ], check=True)
    logger.info("Audio mix complete")


def generate_srt(sentences: list[str], total_duration: float, output_path: str):
    time_per = total_duration / len(sentences)

    def fmt(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w") as f:
        for i, sentence in enumerate(sentences):
            start = i * time_per
            end = (i + 1) * time_per
            f.write(f"{i + 1}\n{fmt(start)} --> {fmt(end)}\n{sentence}\n\n")


def run_with_timeout(fn, job: RenderJob, timeout_seconds: int = 840):
    result = {"error": None}

    def target():
        try:
            fn(job)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.error(f"Render job timed out after {timeout_seconds}s")
        requests.post(job.callback_webhook, json={
            "status": "error",
            "message": f"Render timed out after {timeout_seconds}s"
        })
        return

    if result["error"]:
        raise result["error"]


# ---------------------------------------------------------------------------
# CORE JOB
# ---------------------------------------------------------------------------

def process_video_job(job: RenderJob):
    work_dir = f"/tmp/{uuid.uuid4().hex[:8]}"
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Work dir: {work_dir}")

    try:
        # 1. DOWNLOAD FROM GCS
        logger.info("Downloading assets...")
        gcs_client = storage.Client()
        audio_path = download_from_gcs(job.audio_link, f"{work_dir}/music.mp3", gcs_client)
        video_paths = []
        for i, link in enumerate(job.video_links):
            p = download_from_gcs(link, f"{work_dir}/clip_{i}.mp4", gcs_client)
            video_paths.append(p)

        # 2. STITCH WITH STREAM COPY
        stitched_path = f"{work_dir}/stitched.mp4"
        stitch_videos_concat(video_paths, stitched_path)

        # 3. BUILD AUDIO MIX
        # Uses PCM as intermediate format — no encoding overhead on
        # intermediate steps, only one AAC encode pass at the end.
        mixed_audio_path = f"{work_dir}/mixed_audio.aac"
        mix_audio(
            audio_path, mixed_audio_path,
            job.target_duration, job.hertz_freq,
            job.tone_volume, work_dir
        )

        # 4. GENERATE SRT
        srt_path = f"{work_dir}/subs.srt"
        generate_srt(job.sentences, job.target_duration, srt_path)

        # 5. FINAL RENDER — stream copy, no re-encoding
        logger.info("Rendering final video...")
        final_path = f"{work_dir}/final_meditation.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", stitched_path,
            "-i", mixed_audio_path,
            "-t", str(job.target_duration),
            "-c:v", "copy",
            "-c:a", "copy",
            final_path
        ], check=True)
        logger.info("Render complete")

        # 6. UPLOAD TO GCS
        logger.info("Uploading to GCS...")
        bucket = gcs_client.bucket(os.environ["BUCKET_NAME"])

        blob_name = f"meditation_{uuid.uuid4().hex[:8]}.mp4"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(final_path, content_type="video/mp4",
                                   timeout=300, num_retries=3)

        srt_blob_name = blob_name.replace('.mp4', '.srt')
        srt_blob = bucket.blob(srt_blob_name)
        srt_blob.upload_from_filename(srt_path, content_type="text/plain")

        signed_url = blob.generate_signed_url(expiration=timedelta(hours=24), method="GET")
        srt_signed_url = srt_blob.generate_signed_url(expiration=timedelta(hours=24), method="GET")

        # 7. CALLBACK
        requests.post(job.callback_webhook, json={
            "status": "success",
            "message": f"Render complete ({job.target_duration}s)",
            "video_url": signed_url,
            "srt_url": srt_signed_url
        })
        logger.info(f"Success! Video: {signed_url}")

    except Exception as e:
        logger.error(f"Render failed: {e}", exc_info=True)
        requests.post(job.callback_webhook, json={"status": "error", "message": str(e)})
        raise

    finally:
        subprocess.run(["pkill", "-f", f"ffmpeg.*{work_dir}"], check=False)
        subprocess.run(["rm", "-rf", work_dir], check=False)
        logger.info(f"Cleaned up {work_dir}")


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.post("/render")
async def start_render(job: RenderJob, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_with_timeout, process_video_job, job)
    return {"status": "Processing", "message": "Job accepted. Webhook will fire when complete."}


@app.get("/health")
async def health():
    return {"status": "ok"}