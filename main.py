from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import requests
import os
import uuid
import subprocess
import logging
import numpy as np
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
import moviepy.audio.fx.all as afx
from google.cloud import storage
from datetime import timedelta
import threading

# Structured logging — Cloud Run picks this up automatically
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class RenderJob(BaseModel):
    audio_link: str
    video_links: list[str]
    sentences: list[str]
    callback_webhook: str

    target_duration: int = Field(default=600, description="Total video length in seconds")
    crossfade_time: int = Field(default=3, description="Crossfade transition time in seconds")
    hertz_freq: float = Field(default=432.0, description="Healing frequency tone")
    tone_volume: float = Field(default=0.15, description="Volume of the frequency tone (0.0 to 1.0)")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def download_file_fast(url: str, output_path: str) -> str:
    """
    Direct streaming download — replaces gdown which is interactive-first
    and adds significant overhead in headless server environments.
    """
    logger.info(f"Downloading to {output_path}...")
    if "drive.google.com" in url:
        file_id = url.split('/d/')[1].split('/')[0]
        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                f.write(chunk)
    return output_path


def stitch_videos_ffmpeg(video_paths: list[str], output_path: str, crossfade_sec: int):
    """
    Native ffmpeg xfade filter — replaces MoviePy concatenate_videoclips which
    writes an intermediate temp file and has Python-level overhead per frame.
    Single pass, no intermediate disk write.
    """
    if len(video_paths) == 1:
        subprocess.run(["cp", video_paths[0], output_path], check=True)
        return

    filter_parts = []
    prev_label = "0:v"
    total_offset = 0.0

    for i in range(1, len(video_paths)):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_paths[i - 1]],
            capture_output=True, text=True
        )
        clip_duration = float(probe.stdout.strip())
        total_offset += clip_duration - crossfade_sec
        out_label = f"v{i:02d}"
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={crossfade_sec}"
            f":offset={total_offset:.3f}[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)
    inputs = []
    for p in video_paths:
        inputs += ["-i", p]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_label}]",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",   # fastest for intermediate — final pass uses 'fast'
        "-threads", "0",
        output_path
    ]
    logger.info(f"Stitching {len(video_paths)} clips with ffmpeg xfade...")
    subprocess.run(cmd, check=True)


def generate_srt(sentences: list[str], total_duration: float, output_path: str):
    """
    Write an SRT subtitle file — replaces MoviePy TextClip which shells out to
    ImageMagick per sentence, which is brutally slow at scale. ffmpeg burns the
    SRT in a single pass via the subtitles= filter.
    """
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
    """
    threading.Timer based timeout — signal.SIGALRM won't work here because
    FastAPI runs background tasks in a thread pool, not the main thread.
    """
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
        # Thread is still running — job exceeded timeout
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
        # 1. DOWNLOAD
        logger.info("Downloading assets...")
        audio_path = download_file_fast(job.audio_link, f"{work_dir}/music.mp3")
        video_paths = []
        for i, link in enumerate(job.video_links):
            p = download_file_fast(link, f"{work_dir}/clip_{i}.mp4")
            video_paths.append(p)

        # 2. STITCH WITH FFMPEG (single pass, no intermediate MoviePy write)
        stitched_path = f"{work_dir}/stitched.mp4"
        stitch_videos_ffmpeg(video_paths, stitched_path, job.crossfade_time)

        # 3. BUILD AUDIO MIX
        logger.info("Mixing audio...")
        base_audio = AudioFileClip(audio_path)
        looped_music = afx.audio_loop(base_audio, duration=job.target_duration)
        looped_music = looped_music.set_duration(job.target_duration)

        sample_rate = 44100
        t = np.linspace(0, job.target_duration, int(sample_rate * job.target_duration), endpoint=False)
        tone_wave = (job.tone_volume * np.sin(2 * np.pi * job.hertz_freq * t)).astype(np.float32)
        tone_stereo = np.column_stack([tone_wave, tone_wave])
        hz_clip = AudioArrayClip(tone_stereo, fps=sample_rate)
        hz_clip = hz_clip.set_duration(job.target_duration)

        final_audio = CompositeAudioClip([looped_music, hz_clip])
        final_audio = final_audio.set_duration(job.target_duration)

        mixed_audio_path = f"{work_dir}/mixed_audio.aac"
        final_audio.write_audiofile(mixed_audio_path, fps=sample_rate, codec="aac", logger=None)
        logger.info("Audio mix complete")

        # 4. GENERATE SRT SUBTITLES
        srt_path = f"{work_dir}/subs.srt"
        generate_srt(job.sentences, job.target_duration, srt_path)

        # 5. FINAL FFMPEG PASS: loop video + attach audio + burn subtitles
        logger.info("Rendering final video...")
        final_path = f"{work_dir}/final_meditation.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", stitched_path,
            "-i", mixed_audio_path,
            "-vf", (
                f"subtitles={srt_path}:"
                "force_style='FontName=Helvetica,FontSize=24,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=2,Alignment=2'"
            ),
            "-t", str(job.target_duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "copy",       # audio already encoded above, just remux
            "-threads", "0",
            final_path
        ]
        subprocess.run(cmd, check=True)

        # 6. UPLOAD TO GCS
        logger.info("Uploading to GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.environ["BUCKET_NAME"])
        blob_name = f"meditation_{uuid.uuid4().hex[:8]}.mp4"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(
            final_path,
            content_type="video/mp4",
            timeout=300,
            num_retries=3
        )

        # Signed URL — safer than public bucket, expires in 24h
        signed_url = blob.generate_signed_url(
            expiration=timedelta(hours=24),
            method="GET"
        )

        # 7. CALLBACK
        requests.post(job.callback_webhook, json={
            "status": "success",
            "message": f"Render complete ({job.target_duration}s)",
            "video_url": signed_url
        })
        logger.info(f"Success! Video: {signed_url}")

    except Exception as e:
        logger.error(f"Render failed: {e}", exc_info=True)
        requests.post(job.callback_webhook, json={"status": "error", "message": str(e)})
        raise

    finally:
        # Kill any lingering ffmpeg processes tied to this job before cleaning up
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
    """Cloud Run health check endpoint"""
    return {"status": "ok"}