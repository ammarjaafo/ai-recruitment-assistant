
import os
import cv2
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from tqdm import tqdm
from deepface import DeepFace

def extract_frames(video_path, every_n_seconds=1, output_dir="frames_temp"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_interval = max(int(fps * every_n_seconds), 1)

    frames = []
    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:
            timestamp = frame_id / fps
            frame_path = os.path.join(output_dir, f"frame_{saved_id:04d}.jpg")

            cv2.imwrite(frame_path, frame)

            frames.append({
                "frame_id": saved_id,
                "timestamp": round(timestamp, 2),
                "path": frame_path
            })

            saved_id += 1

        frame_id += 1

    cap.release()

    print(f"Video duration: {duration:.2f} seconds")
    print(f"Extracted frames: {len(frames)}")

    return frames

def extract_audio_from_video(video_path, audio_path="interview_audio.wav"):
    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        raise ValueError("No audio track found in the video.")

    video.audio.write_audiofile(
        audio_path,
        codec="pcm_s16le",
        logger=None
    )

    video.close()

    print("Audio extracted successfully:", audio_path)
    return audio_path

import whisper

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
speech_model = whisper.load_model(WHISPER_MODEL_NAME)

def transcribe_audio(audio_path):
    result = speech_model.transcribe(
        audio_path,
        fp16=False
    )

    transcript = result["text"].strip()
    segments = result.get("segments", [])

    return transcript, segments



def calculate_speech_metrics(transcript, segments):
    words = re.findall(r"\b\w+\b", transcript.lower())
    word_count = len(words)

    if segments:
        duration = segments[-1]["end"] - segments[0]["start"]
    else:
        duration = 0

    duration_minutes = duration / 60 if duration > 0 else 0
    words_per_minute = word_count / duration_minutes if duration_minutes > 0 else 0

    filler_words = {
        "um", "uh", "erm", "ah", "like", "you know", "يعني", "مم", "اه"
    }

    transcript_lower = transcript.lower()

    filler_count = 0
    for filler in filler_words:
        filler_count += transcript_lower.count(filler)

    avg_segment_duration = np.mean(
        [seg["end"] - seg["start"] for seg in segments]
    ) if segments else 0

    return {
        "word_count": int(word_count),
        "speech_duration_seconds": float(round(duration, 2)),
        "words_per_minute": float(round(words_per_minute, 2)),
        "filler_count": int(filler_count),
        "average_segment_duration": float(round(avg_segment_duration, 2)),
        "transcript": transcript
    }

def analyze_pauses(segments):
    if not segments or len(segments) < 2:
        return {
            "pause_count": 0,
            "average_pause_seconds": 0.0,
            "long_pause_count": 0
        }

    pauses = []

    for i in range(1, len(segments)):
        pause = segments[i]["start"] - segments[i - 1]["end"]

        if pause > 0:
            pauses.append(pause)

    long_pauses = [p for p in pauses if p >= 1.5]

    return {
        "pause_count": int(len(pauses)),
        "average_pause_seconds": float(round(np.mean(pauses), 2)) if pauses else 0.0,
        "long_pause_count": int(len(long_pauses))
    }

def summarize_speech_behavior(metrics, pause_metrics):
    wpm = metrics["words_per_minute"]
    filler_count = metrics["filler_count"]
    long_pause_count = pause_metrics["long_pause_count"]

    if 90 <= wpm <= 170:
        pace_level = "Good"
    elif 70 <= wpm < 90 or 170 < wpm <= 190:
        pace_level = "Moderate"
    else:
        pace_level = "Needs Improvement"

    if filler_count <= 2:
        fluency_level = "Good"
    elif filler_count <= 5:
        fluency_level = "Moderate"
    else:
        fluency_level = "Needs Improvement"

    if long_pause_count <= 1:
        pause_control = "Good"
    elif long_pause_count <= 3:
        pause_control = "Moderate"
    else:
        pause_control = "Needs Improvement"

    score_map = {
        "Good": 80,
        "Moderate": 60,
        "Needs Improvement": 40
    }

    speech_score = round(
        score_map[pace_level] * 0.4
        + score_map[fluency_level] * 0.3
        + score_map[pause_control] * 0.3,
        2
    )

    if speech_score >= 75:
        overall_speech = "Strong"
    elif speech_score >= 55:
        overall_speech = "Moderate"
    else:
        overall_speech = "Needs Improvement"

    return {
        "status": "success",
        "pace_level": pace_level,
        "fluency_level": fluency_level,
        "pause_control": pause_control,
        "speech_score": float(speech_score),
        "overall_speech_performance": overall_speech,
        "metrics": metrics,
        "pause_metrics": pause_metrics
    }

def generate_speech_report(summary):
    if summary["status"] != "success":
        return "Speech analysis failed."

    metrics = summary["metrics"]
    pauses = summary["pause_metrics"]

    report = f"""
Speech Behavioral Analysis Report

Overall speech performance:
{summary["overall_speech_performance"]}

Speech score:
{summary["speech_score"]}

Pace level:
{summary["pace_level"]}

Fluency level:
{summary["fluency_level"]}

Pause control:
{summary["pause_control"]}

Speech metrics:
- Word count: {metrics["word_count"]}
- Speech duration: {metrics["speech_duration_seconds"]} seconds
- Words per minute: {metrics["words_per_minute"]}
- Filler words count: {metrics["filler_count"]}
- Pause count: {pauses["pause_count"]}
- Long pause count: {pauses["long_pause_count"]}
- Average pause duration: {pauses["average_pause_seconds"]} seconds

Transcript:
{metrics["transcript"]}

Interpretation:
"""

    if summary["overall_speech_performance"] == "Strong":
        report += "The candidate speaks clearly with good pacing and controlled pauses."
    elif summary["overall_speech_performance"] == "Moderate":
        report += "The candidate shows acceptable speech fluency, with some areas that could be improved."
    else:
        report += "The candidate may need to improve pacing, reduce hesitation, or control long pauses."

    return report

def analyze_frame_emotion(frame_path):
    try:
        result = DeepFace.analyze(
            img_path=frame_path,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "unknown")

        return {
            "success": True,
            "dominant_emotion": dominant_emotion,
            "emotions": emotions
        }

    except Exception as e:
        return {
            "success": False,
            "dominant_emotion": "error",
            "emotions": {},
            "error": str(e)
        }

def analyze_video_emotions(video_path, every_n_seconds=2):
    frames = extract_frames(
        video_path=video_path,
        every_n_seconds=every_n_seconds
    )

    results = []

    for frame in tqdm(frames, desc="Analyzing emotions"):
        analysis = analyze_frame_emotion(frame["path"])

        row = {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "frame_path": frame["path"],
            "success": analysis["success"],
            "dominant_emotion": analysis["dominant_emotion"]
        }

        for emotion, score in analysis["emotions"].items():
            row[emotion] = score

        results.append(row)

    df_results = pd.DataFrame(results)

    return df_results

def summarize_behavioral_video(df_results):
    valid_df = df_results[df_results["success"] == True].copy()

    if valid_df.empty:
        return {
            "status": "failed",
            "message": "No valid frames were analyzed."
        }

    emotion_cols = [
        "angry", "disgust", "fear", "happy",
        "sad", "surprise", "neutral"
    ]

    avg_scores = {}
    for col in emotion_cols:
        if col in valid_df.columns:
            avg_scores[col] = float(round(valid_df[col].mean(), 2))

    emotion_counts = valid_df["dominant_emotion"].value_counts()
    emotion_percentages = (
        emotion_counts / len(valid_df) * 100
    ).round(2).astype(float).to_dict()

    neutral = avg_scores.get("neutral", 0)
    happy = avg_scores.get("happy", 0)
    sad = avg_scores.get("sad", 0)
    fear = avg_scores.get("fear", 0)
    angry = avg_scores.get("angry", 0)

    stress_score = round((fear + sad + angry) / 3, 2)
    expressiveness_score = happy
    calmness_score = round((neutral + (100 - stress_score)) / 2, 2)

    if calmness_score >= 65:
        confidence_indicator = "Medium-High"
    elif calmness_score >= 45:
        confidence_indicator = "Medium"
    else:
        confidence_indicator = "Low"

    if neutral >= 35 and stress_score < 35:
        facial_state = "Neutral / Focused"
    elif happy >= 25:
        facial_state = "Positive / Engaged"
    elif stress_score >= 45:
        facial_state = "Serious / Possibly Nervous"
    else:
        facial_state = "Calm / Serious"

    engagement_score = round(
    (neutral * 0.50)
    + ((100 - stress_score) * 0.35)
    + (happy * 0.15),
    2
)

    if engagement_score >= 50:
        engagement_level = "Good"
    elif engagement_score >= 35:
        engagement_level = "Moderate"
    else:
        engagement_level = "Low"

    summary = {
        "status": "success",
        "total_analyzed_frames": int(len(valid_df)),

        "facial_state": facial_state,
        "confidence_indicator": confidence_indicator,
        "engagement_level": engagement_level,

        "calmness_score": float(calmness_score),
        "stress_score": float(stress_score),
        "expressiveness_score": float(round(expressiveness_score, 2)),
        "engagement_score": float(engagement_score),

        "raw_emotion_distribution_percent": emotion_percentages,
        "raw_average_emotion_scores": avg_scores,

        "note": "Raw emotions are used internally as behavioral signals, not as final psychological judgments."
    }

    return summary

def generate_behavioral_report(summary):
    if summary["status"] != "success":
        return "Behavioral video analysis failed. No valid face frames were detected."

    report = f"""
Behavioral Video Analysis Report

Total analyzed frames:
{summary["total_analyzed_frames"]}

Overall facial state:
{summary["facial_state"]}

Confidence indicator:
{summary["confidence_indicator"]}

Engagement level:
{summary["engagement_level"]}

Behavioral scores:
- Calmness score: {summary["calmness_score"]}
- Stress indicator: {summary["stress_score"]}
- Expressiveness score: {summary["expressiveness_score"]}
- Engagement score: {summary["engagement_score"]}

Interpretation:
"""

    if summary["confidence_indicator"] == "Medium-High":
        report += "The candidate appears generally calm, focused, and visually stable during the interview.\n"
    elif summary["confidence_indicator"] == "Medium":
        report += "The candidate appears reasonably stable, with some signs of seriousness or mild nervousness.\n"
    else:
        report += "The candidate may show visible signs of discomfort, tension, or low facial expressiveness.\n"

    report += """

Important note:
This analysis should be used only as a supportive behavioral insight.
It should not be used as a final hiring decision or as a psychological diagnosis.
"""

    return report

def summarize_temporal_behavior(df_results):
    valid_df = df_results[df_results["success"] == True].copy()

    if valid_df.empty:
        return {
            "status": "failed",
            "message": "No valid frames were analyzed."
        }

    max_time = valid_df["timestamp"].max()

    def get_segment(timestamp):
        if timestamp <= max_time / 3:
            return "Beginning"
        elif timestamp <= (2 * max_time) / 3:
            return "Middle"
        else:
            return "End"

    valid_df["segment"] = valid_df["timestamp"].apply(get_segment)

    segment_reports = {}

    for segment_name, segment_df in valid_df.groupby("segment"):
        emotion_cols = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]

        avg_scores = {}
        for col in emotion_cols:
            if col in segment_df.columns:
                avg_scores[col] = float(round(segment_df[col].mean(), 2))

        neutral = avg_scores.get("neutral", 0)
        happy = avg_scores.get("happy", 0)
        sad = avg_scores.get("sad", 0)
        fear = avg_scores.get("fear", 0)
        angry = avg_scores.get("angry", 0)

        stress_score = round((fear + sad + angry) / 3, 2)
        expressiveness_score = round(happy, 2)
        calmness_score = round((neutral + (100 - stress_score)) / 2, 2)

        engagement_score = round(
            (neutral * 0.50)
            + ((100 - stress_score) * 0.35)
            + (happy * 0.15),
            2
        )

        if calmness_score >= 65:
            confidence_indicator = "Medium-High"
        elif calmness_score >= 45:
            confidence_indicator = "Medium"
        else:
            confidence_indicator = "Low"

        if engagement_score >= 50:
            engagement_level = "Good"
        elif engagement_score >= 35:
            engagement_level = "Moderate"
        else:
            engagement_level = "Low"

        if neutral >= 35 and stress_score < 35:
            facial_state = "Neutral / Focused"
        elif happy >= 25:
            facial_state = "Positive / Engaged"
        elif stress_score >= 45:
            facial_state = "Serious / Possibly Nervous"
        else:
            facial_state = "Calm / Serious"

        segment_reports[segment_name] = {
            "frames": int(len(segment_df)),
            "facial_state": facial_state,
            "confidence_indicator": confidence_indicator,
            "engagement_level": engagement_level,
            "calmness_score": float(calmness_score),
            "stress_score": float(stress_score),
            "expressiveness_score": float(expressiveness_score),
            "engagement_score": float(engagement_score),
            "raw_average_emotion_scores": avg_scores
        }

    return {
        "status": "success",
        "total_analyzed_frames": int(len(valid_df)),
        "duration_seconds": float(round(max_time, 2)),
        "segments": segment_reports
    }

def generate_temporal_behavior_report(summary):
    if summary["status"] != "success":
        return "Temporal behavioral analysis failed."

    ordered_segments = ["Beginning", "Middle", "End"]

    report = f"""
Temporal Behavioral Video Analysis Report

Total analyzed frames:
{summary["total_analyzed_frames"]}

Video duration:
{summary["duration_seconds"]} seconds

Segment Analysis:
"""

    for segment in ordered_segments:
        if segment not in summary["segments"]:
            continue

        data = summary["segments"][segment]

        report += f"""

{segment}:
- Facial state: {data["facial_state"]}
- Confidence indicator: {data["confidence_indicator"]}
- Engagement level: {data["engagement_level"]}
- Calmness score: {data["calmness_score"]}
- Stress indicator: {data["stress_score"]}
- Expressiveness score: {data["expressiveness_score"]}
- Engagement score: {data["engagement_score"]}
"""

    trend_summary = calculate_behavioral_trend(summary)

    if trend_summary["status"] == "success":
        report += f"""

Behavioral Trend:
{trend_summary["trend"]}

Trend Interpretation:
{trend_summary["explanation"]}
"""

    report += """

Overall interpretation:
The report analyzes how the candidate's visual behavior changes over time instead of relying on a single raw emotion label.

Important note:
This analysis should be used only as supportive behavioral insight, not as a final hiring decision.
"""

    return report

def calculate_behavioral_trend(temporal_summary):
    if temporal_summary["status"] != "success":
        return {
            "status": "failed",
            "trend": "Unknown",
            "message": "Temporal summary is not valid."
        }

    segments = temporal_summary["segments"]

    if "Beginning" not in segments or "End" not in segments:
        return {
            "status": "failed",
            "trend": "Unknown",
            "message": "Beginning or End segment is missing."
        }

    begin = segments["Beginning"]
    end = segments["End"]

    begin_score = (
        begin["calmness_score"] * 0.4
        + begin["engagement_score"] * 0.4
        + (100 - begin["stress_score"]) * 0.2
    )

    end_score = (
        end["calmness_score"] * 0.4
        + end["engagement_score"] * 0.4
        + (100 - end["stress_score"]) * 0.2
    )

    change = round(end_score - begin_score, 2)

    if change >= 7:
        trend = "Improving"
        explanation = "The candidate appears more stable or engaged toward the end of the interview."
    elif change <= -7:
        trend = "More Serious Over Time"
        explanation = "The candidate appears less expressive or more serious toward the end of the interview."
    else:
        trend = "Stable"
        explanation = "The candidate maintains relatively consistent visual behavior throughout the interview."

    return {
        "status": "success",
        "trend": trend,
        "beginning_behavior_score": round(begin_score, 2),
        "ending_behavior_score": round(end_score, 2),
        "change": change,
        "explanation": explanation
    }

def plot_behavioral_scores(summary):

    if summary["status"] != "success":
        print("No valid summary to plot.")
        return

    data = {
        "Calmness": summary["calmness_score"],
        "Stress": summary["stress_score"],
        "Expressiveness": summary["expressiveness_score"],
        "Engagement": summary["engagement_score"]
    }

    plt.figure(figsize=(8, 5))

    plt.bar(
        data.keys(),
        data.values()
    )

    plt.title("Behavioral Video Analysis")
    plt.xlabel("Behavioral Indicators")
    plt.ylabel("Score")

    plt.ylim(0, 100)

    plt.show()

def generate_behavioral_trend_report(trend_summary):
    if trend_summary["status"] != "success":
        return "Behavioral trend analysis failed."

    return f"""
Behavioral Trend Analysis

Trend:
{trend_summary["trend"]}

Beginning behavior score:
{trend_summary["beginning_behavior_score"]}

Ending behavior score:
{trend_summary["ending_behavior_score"]}

Change:
{trend_summary["change"]}

Interpretation:
{trend_summary["explanation"]}
"""



def run_behavioral_analysis(video_path, every_n_seconds=2, work_dir=None):
    if work_dir is None:
        work_dir = os.path.dirname(video_path)

    frames_dir = os.path.join(work_dir, "frames")
    audio_path = os.path.join(work_dir, "interview_audio.wav")

    df_emotions = analyze_video_emotions(
        video_path=video_path,
        every_n_seconds=every_n_seconds
    )

    behavior_summary = summarize_behavioral_video(df_emotions)
    behavior_report = generate_behavioral_report(behavior_summary)

    temporal_summary = summarize_temporal_behavior(df_emotions)
    temporal_report = generate_temporal_behavior_report(temporal_summary)

    trend_summary = calculate_behavioral_trend(temporal_summary)

    extract_audio_from_video(video_path, audio_path)
    transcript, segments = transcribe_audio(audio_path)

    speech_metrics = calculate_speech_metrics(transcript, segments)
    pause_metrics = analyze_pauses(segments)
    speech_summary = summarize_speech_behavior(speech_metrics, pause_metrics)
    speech_report = generate_speech_report(speech_summary)

    return {
        "status": "success",
        "video_duration_seconds": temporal_summary.get("duration_seconds"),
        "video_behavior_summary": behavior_summary,
        "temporal_behavior_summary": temporal_summary,
        "behavioral_trend": trend_summary,
        "speech_behavior_summary": speech_summary,
        "reports": {
            "behavior_report": behavior_report,
            "temporal_report": temporal_report,
            "speech_report": speech_report
        },
        "disclaimer": "This analysis provides supportive behavioral insights only and should not be used as a standalone hiring decision."
    }
