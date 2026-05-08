import os
import json
import random
from pathlib import Path

# ----------------------------
# Config
# ----------------------------

video_folder = "" # video path
output_json = "" # output path
seed = 42

random.seed(seed)

# Video extensions to include
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

aot_QA_candidates = [
    {
        "i": "Examine the motion and temporal cues in the video to determine whether it is forward or backward.",
        "q": "Question: Based on the temporal cues, is the video forward or backward?",
        "a_forward": "Answer: forward",
        "a_backward": "Answer: backward"
    },
    {
        "i": "Determine whether the video is played normally or backward based on physical and temporal consistency.",
        "q": "Question: Is this clip more consistent with real-world forward dynamics or backward playback?",
        "a_forward": "Answer: forward",
        "a_backward": "Answer: backward"
    },
    {
        "i": "Identify the arrow of time in the video by deciding whether it runs forward or backward.",
        "q": "Question: What is the arrow of time of this video?",
        "a_forward": "Answer: forward",
        "a_backward": "Answer: backward"
    },
    {
        "i": "Analyze the temporal progression of the video and classify it as forward-time or reversed-time.",
        "q": "Question: Which of the following best describes the temporal direction of this clip?",
        "a_forward": "Answer: forward",
        "a_backward": "Answer: backward"
    },
]

def list_video_files(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    files = sorted(files, key=lambda x: x.name)
    return files

def assign_balanced_aot(video_files):
    """
    Assign ~50% forward and ~50% backward.
    If number of files is odd, one class will have 1 more sample.
    """
    n = len(video_files)
    labels = ["forward"] * (n // 2) + ["backward"] * (n - n // 2)
    random.shuffle(labels)
    return labels

def build_sample(video_name, aot_label):
    qa_template = random.choice(aot_QA_candidates)
    answer = qa_template["a_forward"] if aot_label == "forward" else qa_template["a_backward"]

    return {
        "video": video_name,
        "aot": aot_label,
        "QA": [
            {
                "i": qa_template["i"],
                "q": qa_template["q"],
                "a": answer
            }
        ]
    }

def main():
    video_files = list_video_files(video_folder)

    if not video_files:
        raise ValueError(f"No video files found in: {video_folder}")

    aot_labels = assign_balanced_aot(video_files)

    dataset = []
    for video_path, aot_label in zip(video_files, aot_labels):
        sample = build_sample(video_path.name, aot_label)
        dataset.append(sample)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset)} samples to {output_json}")

    # optional: print class distribution
    num_forward = sum(x["aot"] == "forward" for x in dataset)
    num_backward = sum(x["aot"] == "backward" for x in dataset)
    print(f"forward: {num_forward}, backward: {num_backward}")


if __name__ == "__main__":
    main()