import os
import cv2
import time
import shutil
import tempfile
import subprocess
import streamlit as st
import torch

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import imageio_ffmpeg


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Object Detection and Tracking",
    page_icon="🎯",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f3efff 0%,
            #e7ddf7 50%,
            #f3efff 100%
        );
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    /* Subtitle */
    .main-subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 35px;
    }

    /* Section heading */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-size: 17px;
        font-weight: 700;
        border: none;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
    }

    /* Download button */
    div.stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 700;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 10px;
        border: 1px dashed #cbd5e1;
    }

    /* Progress bar */
    div[data-testid="stProgressBar"] {
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Video */
    video {
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎯 Object Detection & Tracking</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Upload a video and detect objects with bounding boxes, '
    'labels, and tracking IDs'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.45

INFERENCE_SIZE = 480

FRAME_SKIP = 2

MAX_DIMENSION = 640


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")


# --------------------------------------------------
# CHECK FOR POSSIBLE AUTO-RICKSHAW
# --------------------------------------------------

def is_possible_auto_rickshaw(
    box,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = box

    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )

    aspect_ratio = (
        box_width / box_height
    )

    relative_width = (
        box_width / frame_width
    )

    relative_height = (
        box_height / frame_height
    )

    if (
        0.55 <= aspect_ratio <= 2.20
        and 0.03 <= relative_width <= 0.45
        and 0.04 <= relative_height <= 0.50
    ):
        return True

    return False


# --------------------------------------------------
# DRAW DETECTION
# --------------------------------------------------

def draw_detection(
    frame,
    box,
    class_name,
    confidence,
    track_id
):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(
            x1,
            width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            height - 1
        )
    )

    x2 = max(
        0,
        min(
            x2,
            width - 1
        )
    )

    y2 = max(
        0,
        min(
            y2,
            height - 1
        )
    )

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    if track_id is not None:

        label = (
            f"{class_name} | "
            f"ID: {track_id} | "
            f"{confidence:.2f}"
        )

    else:

        label = (
            f"{class_name} | "
            f"{confidence:.2f}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    label_top = max(
        0,
        y1 - text_height - baseline - 8
    )

    label_bottom = min(
        height - 1,
        label_top +
        text_height +
        baseline +
        8
    )

    label_right = min(
        width - 1,
        x1 +
        text_width +
        8
    )

    cv2.rectangle(
        frame,
        (x1, label_top),
        (label_right, label_bottom),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            label_top + text_height + 2
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# --------------------------------------------------
# IOU
# --------------------------------------------------

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    intersection_width = max(
        0,
        ix2 - ix1
    )

    intersection_height = max(
        0,
        iy2 - iy1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area_a = (
        max(
            0,
            ax2 - ax1
        ) *
        max(
            0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0,
            bx2 - bx1
        ) *
        max(
            0,
            by2 - by1
        )
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

def process_video(
    input_path,
    output_path
):

    model = load_model()

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=100
    )

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open the uploaded video."
        )

    writer = None

    try:

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:
            fps = 25.0

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if width <= 0 or height <= 0:

            raise RuntimeError(
                "The uploaded video has invalid dimensions."
            )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (width, height)
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create the processed video."
            )

        progress_bar = st.progress(0)

        status_text = st.empty()

        frame_number = 0

        total_detections = 0

        start_time = time.time()

        last_detections = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            should_detect = (
                frame_number == 1
                or frame_number % FRAME_SKIP == 0
            )

            if should_detect:

                original_height, original_width = (
                    frame.shape[:2]
                )

                scale = min(
                    1.0,
                    MAX_DIMENSION /
                    max(
                        original_width,
                        original_height
                    )
                )

                if scale < 1.0:

                    inference_frame = cv2.resize(
                        frame,
                        (
                            max(
                                1,
                                int(
                                    original_width *
                                    scale
                                )
                            ),
                            max(
                                1,
                                int(
                                    original_height *
                                    scale
                                )
                            )
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                else:

                    inference_frame = frame

                results = model.predict(
                    inference_frame,
                    imgsz=INFERENCE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=0.50,
                    device=device,
                    verbose=False
                )

                detections = []

                for result in results:

                    if result.boxes is None:
                        continue

                    for box in result.boxes:

                        confidence = float(
                            box.conf[0].item()
                        )

                        if (
                            confidence <
                            CONFIDENCE_THRESHOLD
                        ):
                            continue

                        class_id = int(
                            box.cls[0].item()
                        )

                        class_name = str(
                            model.names[class_id]
                        )

                        coords = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        x1, y1, x2, y2 = [
                            int(
                                value / scale
                            )
                            for value in coords
                        ]

                        x1 = max(
                            0,
                            min(
                                x1,
                                original_width - 1
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                y1,
                                original_height - 1
                            )
                        )

                        x2 = max(
                            0,
                            min(
                                x2,
                                original_width - 1
                            )
                        )

                        y2 = max(
                            0,
                            min(
                                y2,
                                original_height - 1
                            )
                        )

                        if (
                            x2 <= x1
                            or y2 <= y1
                        ):
                            continue

                        display_class = class_name

                        if class_name == "truck":

                            possible_auto = (
                                is_possible_auto_rickshaw(
                                    (
                                        x1,
                                        y1,
                                        x2,
                                        y2
                                    ),
                                    original_width,
                                    original_height
                                )
                            )

                            if possible_auto:

                                display_class = (
                                    "Auto Rickshaw"
                                )

                        detections.append(
                            {
                                "box": (
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ),
                                "confidence": confidence,
                                "class_name": display_class,
                                "tracker_class": class_name,
                                "track_id": None
                            }
                        )

                tracker_detections = []

                for detection in detections:

                    x1, y1, x2, y2 = (
                        detection["box"]
                    )

                    tracker_detections.append(
                        (
                            [
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1
                            ],
                            detection[
                                "confidence"
                            ],
                            detection[
                                "tracker_class"
                            ]
                        )
                    )

                tracks = tracker.update_tracks(
                    tracker_detections,
                    frame=frame
                )

                used_detections = set()

                for track in tracks:

                    if not track.is_confirmed():
                        continue

                    track_box = tuple(
                        int(value)
                        for value in
                        track.to_ltrb()
                    )

                    best_index = None

                    best_iou = 0.0

                    for index, detection in enumerate(
                        detections
                    ):

                        if (
                            index in
                            used_detections
                        ):
                            continue

                        detected_class = (
                            detection[
                                "tracker_class"
                            ]
                        )

                        tracked_class = (
                            track.get_det_class()
                        )

                        if (
                            tracked_class
                            and
                            detected_class
                            != tracked_class
                        ):
                            continue

                        overlap = calculate_iou(
                            track_box,
                            detection["box"]
                        )

                        if overlap > best_iou:

                            best_iou = overlap

                            best_index = index

                    if (
                        best_index is not None
                        and best_iou >= 0.20
                    ):

                        detections[
                            best_index
                        ]["track_id"] = (
                            track.track_id
                        )

                        used_detections.add(
                            best_index
                        )

                last_detections = []

                for detection in detections:

                    last_detections.append(
                        {
                            "box": detection[
                                "box"
                            ],
                            "confidence": detection[
                                "confidence"
                            ],
                            "class_name": detection[
                                "class_name"
                            ],
                            "track_id": detection[
                                "track_id"
                            ]
                        }
                    )

                total_detections += len(
                    detections
                )

            else:

                detections = last_detections

            for detection in detections:

                draw_detection(
                    frame,
                    detection["box"],
                    detection["class_name"],
                    detection["confidence"],
                    detection["track_id"]
                )

            writer.write(
                frame
            )

            if total_frames > 0:

                progress = (
                    frame_number /
                    total_frames
                )

                progress = min(
                    max(progress, 0.0),
                    1.0
                )

                progress_bar.progress(
                    progress
                )

            status_text.text(
                f"Processing frame "
                f"{frame_number}"
                f" of {total_frames}"
                f" | Objects detected: "
                f"{total_detections}"
            )

        if frame_number == 0:

            raise RuntimeError(
                "No readable frames were found "
                "in the video."
            )

        return {
            "frames": frame_number,
            "detections": total_detections,
            "elapsed": (
                time.time() -
                start_time
            ),
            "progress_bar": progress_bar,
            "status_text": status_text
        }

    finally:

        cap.release()

        if writer is not None:
            writer.release()


# --------------------------------------------------
# CONVERT VIDEO FOR BROWSER PLAYBACK
# --------------------------------------------------

def convert_for_browser(
    input_path,
    output_path
):

    ffmpeg_exe = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg_exe:

        ffmpeg_exe = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600
        )

        return (
            result.returncode == 0
            and os.path.exists(
                output_path
            )
            and os.path.getsize(
                output_path
            ) > 0
        )

    except (
        OSError,
        subprocess.TimeoutExpired
    ):

        return False


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📁 Upload Video</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# --------------------------------------------------
# START DETECTION AND TRACKING
# --------------------------------------------------

if uploaded_file is not None:

    st.write(
        f"**Selected video:** {uploaded_file.name}"
    )

    st.write(
        "Ready to detect and track objects."
    )

    if st.button(
        "🎯 Start Detection and Tracking",
        type="primary"
    ):

        input_path = None
        output_path = None
        converted_path = None

        try:

            # --------------------------------------------------
            # SAVE UPLOADED VIDEO
            # --------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=(
                    os.path.splitext(
                        uploaded_file.name
                    )[1]
                    or ".mp4"
                )
            ) as temp_input:

                temp_input.write(
                    uploaded_file.getbuffer()
                )

                input_path = (
                    temp_input.name
                )

            # --------------------------------------------------
            # OUTPUT FILE
            # --------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_output:

                output_path = (
                    temp_output.name
                )

            # --------------------------------------------------
            # BROWSER VIDEO
            # --------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_converted:

                converted_path = (
                    temp_converted.name
                )

            # --------------------------------------------------
            # PROCESS VIDEO
            # --------------------------------------------------

            stats = process_video(
                input_path,
                output_path
            )

            # --------------------------------------------------
            # CONVERT VIDEO
            # --------------------------------------------------

            converted = (
                convert_for_browser(
                    output_path,
                    converted_path
                )
            )

            if converted:

                final_video_path = (
                    converted_path
                )

            else:

                final_video_path = (
                    output_path
                )

            # --------------------------------------------------
            # READ VIDEO
            # --------------------------------------------------

            with open(
                final_video_path,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            # --------------------------------------------------
            # COMPLETION
            # --------------------------------------------------

            stats[
                "progress_bar"
            ].progress(1.0)

            stats[
                "status_text"
            ].success(
                "Detection and tracking completed "
                f"in {stats['elapsed']:.1f} seconds!"
            )

            # --------------------------------------------------
            # PROCESSED VIDEO
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                '🎬 Processed Video'
                '</div>',
                unsafe_allow_html=True
            )

            st.video(
                video_bytes,
                format="video/mp4"
            )

            # --------------------------------------------------
            # DOWNLOAD
            # --------------------------------------------------

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )

        finally:

            # --------------------------------------------------
            # CLEAN TEMPORARY FILES
            # --------------------------------------------------

            for path in (
                input_path,
                output_path,
                converted_path
            ):

                if (
                    path
                    and os.path.exists(path)
                ):

                    try:

                        os.remove(path)

                    except OSError:

                        pass

else:

    st.info(
        "Upload a recorded video to get started."
    )