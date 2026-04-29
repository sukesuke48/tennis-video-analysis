import csv
import math
import tempfile
import traceback
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


STATIC_FRAME_COUNT = 60

RESULT_VIDEO_NAME = "skeleton_video.mp4"
RESULT_CSV_NAME = "analysis.csv"
RESULT_GRAPH_NAME = "graphs.png"
STATIC_FRAMES_DIR_NAME = "static_frames"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def sanitize_windows_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    safe = "".join("_" if ch in invalid_chars else ch for ch in str(name))
    safe = safe.strip().strip(".")
    if safe == "":
        safe = "video"

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    if safe.upper() in reserved:
        safe = f"_{safe}"

    return safe


def open_video_or_error(video_path, error_text):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(error_text)
    return cap


def safe_release(cap):
    if cap is not None:
        cap.release()


def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frame_count


def read_frame_at(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def build_even_indices(frame_count: int, count: int):
    if frame_count <= 0:
        return []

    count = max(1, min(count, frame_count))
    if count == 1:
        return [0]

    indices = []
    for i in range(count):
        ratio = i / (count - 1)
        idx = int(round((frame_count - 1) * ratio))
        indices.append(idx)

    return sorted(set(indices))


def get_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text_unicode(frame, lines, x, y, font_size=28, fill=(255, 255, 255), stroke_fill=(0, 0, 0), line_gap=8):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = get_font(font_size)

    yy = y
    for line in lines:
        draw.text(
            (x, yy),
            str(line),
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=stroke_fill,
        )
        try:
            bbox = draw.textbbox((x, yy), str(line), font=font, stroke_width=2)
            line_h = bbox[3] - bbox[1]
        except Exception:
            line_h = font_size
        yy += line_h + line_gap

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = bgr


def save_image_unicode(image, path):
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == "":
        ext = ".png"

    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False

    encoded.tofile(path)
    return True


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)


def calculate_line_angle(p1, p2):
    if p1 is None or p2 is None:
        return None

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0 and dy == 0:
        return None

    angle_deg = np.degrees(np.arctan2(dy, dx))
    return float(angle_deg)


def safe_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_angle_deg(angle):
    if angle is None:
        return None
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return float(angle)


def distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)))


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def to_pixel_point(norm_xy, width, height):
    if norm_xy is None:
        return None
    x = int(norm_xy[0] * width)
    y = int(norm_xy[1] * height)
    return (x, y)


def point_to_tuple_float(p):
    if p is None:
        return None
    return (float(p[0]), float(p[1]))


def relative_point(point, origin):
    if point is None or origin is None:
        return None
    return (float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1]))


def get_xy(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return float(lm.x), float(lm.y)


def get_xyz_visibility(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)


def choose_active_arm(left_wrist_px, right_wrist_px, left_elbow_px, right_elbow_px, width, height):
    frame_center = (width / 2.0, height / 2.0)

    left_score = -1e9
    right_score = -1e9

    if left_wrist_px is not None and left_elbow_px is not None:
        left_len = distance(left_wrist_px, left_elbow_px)
        left_center_dist = distance(left_wrist_px, frame_center)
        left_score = left_len * 1.5 + left_center_dist

    if right_wrist_px is not None and right_elbow_px is not None:
        right_len = distance(right_wrist_px, right_elbow_px)
        right_center_dist = distance(right_wrist_px, frame_center)
        right_score = right_len * 1.5 + right_center_dist

    if right_score >= left_score:
        return "right"
    return "left"


def detect_best_shaft_line(frame, wrist_px, elbow_px):
    if wrist_px is None or elbow_px is None:
        return None

    h, w = frame.shape[:2]

    arm_len = distance(wrist_px, elbow_px)
    if arm_len is None:
        return None

    roi_half = int(max(120, min(320, arm_len * 2.6)))
    x1 = clamp(wrist_px[0] - roi_half, 0, w - 1)
    y1 = clamp(wrist_px[1] - roi_half, 0, h - 1)
    x2 = clamp(wrist_px[0] + roi_half, 0, w - 1)
    y2 = clamp(wrist_px[1] + roi_half, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)

    min_line_length = int(max(50, arm_len * 0.9))
    max_line_gap = int(max(12, arm_len * 0.25))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return None

    best_line = None
    best_score = -1e18

    wrist_np = np.array(wrist_px, dtype=np.float64)
    elbow_np = np.array(elbow_px, dtype=np.float64)
    forearm_vec = wrist_np - elbow_np
    forearm_norm = np.linalg.norm(forearm_vec)

    for line in lines[:, 0]:
        lx1, ly1, lx2, ly2 = line
        p1 = np.array([lx1 + x1, ly1 + y1], dtype=np.float64)
        p2 = np.array([lx2 + x1, ly2 + y1], dtype=np.float64)

        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < min_line_length:
            continue

        d1 = np.linalg.norm(p1 - wrist_np)
        d2 = np.linalg.norm(p2 - wrist_np)
        min_wrist_dist = min(d1, d2)

        if forearm_norm > 0 and line_len > 0:
            cos_sim = np.dot(line_vec, forearm_vec) / (line_len * forearm_norm)
            cos_sim = abs(float(cos_sim))
        else:
            cos_sim = 0.0

        midpoint = (p1 + p2) / 2.0
        midpoint_dist = np.linalg.norm(midpoint - wrist_np)

        score = (
            line_len * 2.2
            - min_wrist_dist * 1.8
            - midpoint_dist * 0.25
            + cos_sim * 120.0
        )

        if score > best_score:
            best_score = score
            best_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))

    return best_line


def detect_racket_head_region(frame, head_hint_px, butt_px, search_radius):
    if head_hint_px is None or butt_px is None:
        return None, None, None

    h, w = frame.shape[:2]
    r = int(max(70, min(220, search_radius)))

    x1 = clamp(int(head_hint_px[0] - r), 0, w - 1)
    y1 = clamp(int(head_hint_px[1] - r), 0, h - 1)
    x2 = clamp(int(head_hint_px[0] + r), 0, w - 1)
    y2 = clamp(int(head_hint_px[1] + r), 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_angle = None
    best_box_global = None
    best_score = -1e18

    hint_np = np.array(head_hint_px, dtype=np.float64)
    butt_np = np.array(butt_px, dtype=np.float64)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect

        long_side = max(rw, rh)
        short_side = min(rw, rh)

        if long_side < 20:
            continue

        center_global = np.array([cx + x1, cy + y1], dtype=np.float64)
        center_dist = np.linalg.norm(center_global - hint_np)

        if short_side == 0:
            aspect_ratio = 999.0
        else:
            aspect_ratio = long_side / short_side

        box = cv2.boxPoints(rect)
        box_global = box + np.array([x1, y1], dtype=np.float32)

        farthest_point = None
        farthest_dist = -1.0
        for pt in box_global:
            d = np.linalg.norm(pt - butt_np)
            if d > farthest_dist:
                farthest_dist = d
                farthest_point = pt

        score = area * 0.03 - center_dist * 1.4 - aspect_ratio * 8.0 + farthest_dist * 0.15

        if score > best_score:
            best_score = score
            best_center = (float(center_global[0]), float(center_global[1]))
            best_angle = calculate_line_angle(
                (float(center_global[0]), float(center_global[1])),
                (float(farthest_point[0]), float(farthest_point[1])),
            )
            best_box_global = box_global

    return best_center, normalize_angle_deg(best_angle), best_box_global


def project_extreme(points, origin, axis_unit, use_max):
    values = []
    for p in points:
        v = np.dot(np.array(p, dtype=np.float64) - np.array(origin, dtype=np.float64), axis_unit)
        values.append(v)
    idx = int(np.argmax(values) if use_max else np.argmin(values))
    p = points[idx]
    return (float(p[0]), float(p[1]))


def build_racket_five_points(butt_px, head_hint_px, center_px, box_global):
    if butt_px is None or center_px is None:
        return None, None, None, None, None, None

    butt_np = np.array(butt_px, dtype=np.float64)
    center_np = np.array(center_px, dtype=np.float64)

    shaft_vec = center_np - butt_np
    shaft_norm = np.linalg.norm(shaft_vec)

    if shaft_norm == 0:
        return None, None, None, None, None, None

    shaft_unit = shaft_vec / shaft_norm
    face_unit = np.array([-shaft_unit[1], shaft_unit[0]], dtype=np.float64)

    if box_global is not None and len(box_global) >= 4:
        pts = [tuple(map(float, p)) for p in box_global]
        top_pt = project_extreme(pts, butt_px, shaft_unit, True)
        left_pt = project_extreme(pts, center_px, face_unit, False)
        right_pt = project_extreme(pts, center_px, face_unit, True)
    else:
        approx_half_width = max(10.0, shaft_norm * 0.18)
        top_pt = (
            float(center_np[0] + shaft_unit[0] * approx_half_width),
            float(center_np[1] + shaft_unit[1] * approx_half_width),
        )
        left_pt = (
            float(center_np[0] - face_unit[0] * approx_half_width),
            float(center_np[1] - face_unit[1] * approx_half_width),
        )
        right_pt = (
            float(center_np[0] + face_unit[0] * approx_half_width),
            float(center_np[1] + face_unit[1] * approx_half_width),
        )

    center_pt = (float(center_px[0]), float(center_px[1]))
    grip_end_pt = (float(butt_px[0]), float(butt_px[1]))
    face_angle_deg = normalize_angle_deg(calculate_line_angle(left_pt, right_pt))

    return top_pt, left_pt, center_pt, right_pt, grip_end_pt, face_angle_deg


def detect_tennis_racket_five_points(frame, left_wrist_px, right_wrist_px, left_elbow_px, right_elbow_px):
    h, w = frame.shape[:2]

    active_side = choose_active_arm(
        left_wrist_px,
        right_wrist_px,
        left_elbow_px,
        right_elbow_px,
        w,
        h,
    )

    if active_side == "right":
        wrist_px = right_wrist_px
        elbow_px = right_elbow_px
    else:
        wrist_px = left_wrist_px
        elbow_px = left_elbow_px

    if wrist_px is None or elbow_px is None:
        return None

    best_line = detect_best_shaft_line(frame, wrist_px, elbow_px)
    if best_line is None:
        return None

    p1, p2 = best_line
    d1 = distance(p1, wrist_px)
    d2 = distance(p2, wrist_px)

    if d1 <= d2:
        butt_px = point_to_tuple_float(p1)
        head_hint_px = point_to_tuple_float(p2)
    else:
        butt_px = point_to_tuple_float(p2)
        head_hint_px = point_to_tuple_float(p1)

    shaft_len = distance(butt_px, head_hint_px)
    if shaft_len is None:
        return None

    center_px, rough_angle_deg, box_global = detect_racket_head_region(
        frame,
        head_hint_px,
        butt_px,
        search_radius=shaft_len * 0.9,
    )

    if center_px is None:
        center_px = head_hint_px

    top_pt, left_pt, center_pt, right_pt, grip_end_pt, face_angle_deg = build_racket_five_points(
        butt_px,
        head_hint_px,
        center_px,
        box_global,
    )

    if center_pt is None:
        return None

    result = {
        "active_side": active_side,
        "shaft_head_hint_x": head_hint_px[0] if head_hint_px is not None else None,
        "shaft_head_hint_y": head_hint_px[1] if head_hint_px is not None else None,
        "racket_top_x": top_pt[0] if top_pt is not None else None,
        "racket_top_y": top_pt[1] if top_pt is not None else None,
        "racket_left_x": left_pt[0] if left_pt is not None else None,
        "racket_left_y": left_pt[1] if left_pt is not None else None,
        "racket_center_x": center_pt[0] if center_pt is not None else None,
        "racket_center_y": center_pt[1] if center_pt is not None else None,
        "racket_right_x": right_pt[0] if right_pt is not None else None,
        "racket_right_y": right_pt[1] if right_pt is not None else None,
        "racket_grip_end_x": grip_end_pt[0] if grip_end_pt is not None else None,
        "racket_grip_end_y": grip_end_pt[1] if grip_end_pt is not None else None,
        "racket_face_angle_deg": face_angle_deg,
        "racket_shaft_angle_deg": normalize_angle_deg(calculate_line_angle(grip_end_pt, top_pt)),
        "racket_head_box_detected": 1 if box_global is not None else 0,
    }

    if left_pt is not None and right_pt is not None:
        result["racket_face_width_px"] = distance(left_pt, right_pt)
    else:
        result["racket_face_width_px"] = None

    if top_pt is not None and grip_end_pt is not None:
        result["racket_length_px"] = distance(top_pt, grip_end_pt)
    else:
        result["racket_length_px"] = None

    if left_pt is not None and center_pt is not None and grip_end_pt is not None:
        result["racket_face_to_shaft_angle_deg"] = calculate_angle(left_pt, center_pt, grip_end_pt)
    else:
        result["racket_face_to_shaft_angle_deg"] = None

    return result


def draw_racket_five_points(frame, row):
    point_specs = [
        ("racket_top", "TOP", (0, 0, 255)),
        ("racket_left", "LEFT", (0, 255, 255)),
        ("racket_center", "CENTER", (255, 255, 0)),
        ("racket_right", "RIGHT", (255, 0, 255)),
        ("racket_grip_end", "GRIP", (0, 255, 0)),
    ]

    point_map = {}

    for key, label, color in point_specs:
        x = row.get(f"{key}_x")
        y = row.get(f"{key}_y")
        if x is None or y is None:
            continue

        p = (int(round(float(x))), int(round(float(y))))
        point_map[key] = p
        cv2.circle(frame, p, 6, color, -1)
        cv2.putText(
            frame,
            label,
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    if "racket_grip_end" in point_map and "racket_top" in point_map:
        cv2.line(frame, point_map["racket_grip_end"], point_map["racket_top"], (255, 255, 0), 2)

    if "racket_left" in point_map and "racket_right" in point_map:
        cv2.line(frame, point_map["racket_left"], point_map["racket_right"], (255, 255, 255), 2)

    if "racket_center" in point_map and row.get("racket_face_angle_deg") is not None:
        c = point_map["racket_center"]
        angle_deg = float(row["racket_face_angle_deg"])
        length = 70
        rad = np.radians(angle_deg)
        end_x = int(c[0] + length * np.cos(rad))
        end_y = int(c[1] + length * np.sin(rad))
        cv2.arrowedLine(frame, c, (end_x, end_y), (0, 255, 0), 2, tipLength=0.2)


def add_relative_landmark_fields(row, name, px_xy, base_px_xy):
    if px_xy is None or base_px_xy is None:
        row[f"{name}_base_right_heel_px_x"] = None
        row[f"{name}_base_right_heel_px_y"] = None
    else:
        row[f"{name}_base_right_heel_px_x"] = float(px_xy[0] - base_px_xy[0])
        row[f"{name}_base_right_heel_px_y"] = float(px_xy[1] - base_px_xy[1])


def extract_pose_row_data(landmarks, width, height):
    row = {}

    left_shoulder = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_elbow = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
    right_elbow = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
    left_wrist = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
    right_wrist = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
    left_index = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_INDEX)
    right_index = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_INDEX)
    left_hip = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    left_knee = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
    right_knee = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
    left_ankle = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    right_ankle = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
    left_heel = get_xy(landmarks, mp_pose.PoseLandmark.LEFT_HEEL)
    right_heel = get_xy(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL)
    nose = get_xy(landmarks, mp_pose.PoseLandmark.NOSE)

    left_wrist_px = to_pixel_point(left_wrist, width, height)
    right_wrist_px = to_pixel_point(right_wrist, width, height)
    left_elbow_px = to_pixel_point(left_elbow, width, height)
    right_elbow_px = to_pixel_point(right_elbow, width, height)
    right_heel_px = to_pixel_point(right_heel, width, height)

    landmark_names = [
        ("nose", mp_pose.PoseLandmark.NOSE),
        ("left_shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
        ("right_shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
        ("left_elbow", mp_pose.PoseLandmark.LEFT_ELBOW),
        ("right_elbow", mp_pose.PoseLandmark.RIGHT_ELBOW),
        ("left_wrist", mp_pose.PoseLandmark.LEFT_WRIST),
        ("right_wrist", mp_pose.PoseLandmark.RIGHT_WRIST),
        ("left_index", mp_pose.PoseLandmark.LEFT_INDEX),
        ("right_index", mp_pose.PoseLandmark.RIGHT_INDEX),
        ("left_hip", mp_pose.PoseLandmark.LEFT_HIP),
        ("right_hip", mp_pose.PoseLandmark.RIGHT_HIP),
        ("left_knee", mp_pose.PoseLandmark.LEFT_KNEE),
        ("right_knee", mp_pose.PoseLandmark.RIGHT_KNEE),
        ("left_ankle", mp_pose.PoseLandmark.LEFT_ANKLE),
        ("right_ankle", mp_pose.PoseLandmark.RIGHT_ANKLE),
        ("left_heel", mp_pose.PoseLandmark.LEFT_HEEL),
        ("right_heel", mp_pose.PoseLandmark.RIGHT_HEEL),
    ]

    landmark_px_map = {}

    for name, enum_value in landmark_names:
        x, y, z, visibility = get_xyz_visibility(landmarks, enum_value)
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
        row[f"{name}_z"] = z
        row[f"{name}_visibility"] = visibility
        landmark_px_map[name] = to_pixel_point((x, y), width, height)

    for name, _ in landmark_names:
        add_relative_landmark_fields(
            row=row,
            name=name,
            px_xy=landmark_px_map[name],
            base_px_xy=right_heel_px,
        )

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_palm_angle_deg = calculate_line_angle(left_wrist, left_index)
    right_palm_angle_deg = calculate_line_angle(right_wrist, right_index)

    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_center = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
    )

    trunk_angle_deg = calculate_line_angle(hip_center, shoulder_center)
    shoulder_line_angle_deg = calculate_line_angle(left_shoulder, right_shoulder)
    hip_line_angle_deg = calculate_line_angle(left_hip, right_hip)
    gaze_angle_deg = calculate_line_angle(shoulder_center, nose)

    shoulder_center_px = (shoulder_center[0] * width, shoulder_center[1] * height)
    hip_center_px = (hip_center[0] * width, hip_center[1] * height)
    shoulder_center_rel_px = relative_point(shoulder_center_px, right_heel_px)
    hip_center_rel_px = relative_point(hip_center_px, right_heel_px)

    row["pose_visible"] = 1
    row["position_base_name"] = "right_heel"
    row["base_right_heel_px_x"] = 0.0 if right_heel_px is not None else None
    row["base_right_heel_px_y"] = 0.0 if right_heel_px is not None else None
    row["left_elbow_angle_deg"] = left_elbow_angle
    row["right_elbow_angle_deg"] = right_elbow_angle
    row["left_shoulder_angle_deg"] = left_shoulder_angle
    row["right_shoulder_angle_deg"] = right_shoulder_angle
    row["left_hip_angle_deg"] = left_hip_angle
    row["right_hip_angle_deg"] = right_hip_angle
    row["left_knee_angle_deg"] = left_knee_angle
    row["right_knee_angle_deg"] = right_knee_angle
    row["left_ankle_angle_deg"] = None
    row["right_ankle_angle_deg"] = None
    row["left_palm_angle_deg"] = left_palm_angle_deg
    row["right_palm_angle_deg"] = right_palm_angle_deg
    row["trunk_angle_deg"] = trunk_angle_deg
    row["shoulder_line_angle_deg"] = shoulder_line_angle_deg
    row["hip_line_angle_deg"] = hip_line_angle_deg
    row["gaze_angle_deg"] = gaze_angle_deg
    row["hip_center_x"] = hip_center_px[0]
    row["hip_center_y"] = hip_center_px[1]
    row["shoulder_center_x"] = shoulder_center_px[0]
    row["shoulder_center_y"] = shoulder_center_px[1]
    row["hip_center_base_right_heel_px_x"] = hip_center_rel_px[0] if hip_center_rel_px is not None else None
    row["hip_center_base_right_heel_px_y"] = hip_center_rel_px[1] if hip_center_rel_px is not None else None
    row["shoulder_center_base_right_heel_px_x"] = shoulder_center_rel_px[0] if shoulder_center_rel_px is not None else None
    row["shoulder_center_base_right_heel_px_y"] = shoulder_center_rel_px[1] if shoulder_center_rel_px is not None else None
    row["avg_knee_angle_deg"] = float(np.nanmean([safe_float(left_knee_angle), safe_float(right_knee_angle)]))

    row["left_wrist_px_x"] = left_wrist_px[0] if left_wrist_px is not None else None
    row["left_wrist_px_y"] = left_wrist_px[1] if left_wrist_px is not None else None
    row["right_wrist_px_x"] = right_wrist_px[0] if right_wrist_px is not None else None
    row["right_wrist_px_y"] = right_wrist_px[1] if right_wrist_px is not None else None
    row["left_elbow_px_x"] = left_elbow_px[0] if left_elbow_px is not None else None
    row["left_elbow_px_y"] = left_elbow_px[1] if left_elbow_px is not None else None
    row["right_elbow_px_x"] = right_elbow_px[0] if right_elbow_px is not None else None
    row["right_elbow_px_y"] = right_elbow_px[1] if right_elbow_px is not None else None

    return row, left_wrist_px, right_wrist_px, left_elbow_px, right_elbow_px


def add_empty_pose_fields(row):
    keys = [
        "pose_visible",
        "position_base_name",
        "base_right_heel_px_x", "base_right_heel_px_y",
        "left_elbow_angle_deg", "right_elbow_angle_deg",
        "left_shoulder_angle_deg", "right_shoulder_angle_deg",
        "left_hip_angle_deg", "right_hip_angle_deg",
        "left_knee_angle_deg", "right_knee_angle_deg",
        "left_ankle_angle_deg", "right_ankle_angle_deg",
        "left_palm_angle_deg", "right_palm_angle_deg",
        "trunk_angle_deg", "shoulder_line_angle_deg",
        "hip_line_angle_deg", "gaze_angle_deg",
        "hip_center_x", "hip_center_y",
        "shoulder_center_x", "shoulder_center_y",
        "hip_center_base_right_heel_px_x", "hip_center_base_right_heel_px_y",
        "shoulder_center_base_right_heel_px_x", "shoulder_center_base_right_heel_px_y",
        "avg_knee_angle_deg",
        "left_wrist_px_x", "left_wrist_px_y",
        "right_wrist_px_x", "right_wrist_px_y",
        "left_elbow_px_x", "left_elbow_px_y",
        "right_elbow_px_x", "right_elbow_px_y",
        "nose_base_right_heel_px_x", "nose_base_right_heel_px_y",
        "left_shoulder_base_right_heel_px_x", "left_shoulder_base_right_heel_px_y",
        "right_shoulder_base_right_heel_px_x", "right_shoulder_base_right_heel_px_y",
        "left_elbow_base_right_heel_px_x", "left_elbow_base_right_heel_px_y",
        "right_elbow_base_right_heel_px_x", "right_elbow_base_right_heel_px_y",
        "left_wrist_base_right_heel_px_x", "left_wrist_base_right_heel_px_y",
        "right_wrist_base_right_heel_px_x", "right_wrist_base_right_heel_px_y",
        "left_index_base_right_heel_px_x", "left_index_base_right_heel_px_y",
        "right_index_base_right_heel_px_x", "right_index_base_right_heel_px_y",
        "left_hip_base_right_heel_px_x", "left_hip_base_right_heel_px_y",
        "right_hip_base_right_heel_px_x", "right_hip_base_right_heel_px_y",
        "left_knee_base_right_heel_px_x", "left_knee_base_right_heel_px_y",
        "right_knee_base_right_heel_px_x", "right_knee_base_right_heel_px_y",
        "left_ankle_base_right_heel_px_x", "left_ankle_base_right_heel_px_y",
        "right_ankle_base_right_heel_px_x", "right_ankle_base_right_heel_px_y",
        "left_heel_base_right_heel_px_x", "left_heel_base_right_heel_px_y",
        "right_heel_base_right_heel_px_x", "right_heel_base_right_heel_px_y",
    ]
    row["pose_visible"] = 0
    for key in keys:
        if key != "pose_visible":
            row[key] = None


def add_empty_racket_fields(row):
    keys = [
        "active_side",
        "shaft_head_hint_x", "shaft_head_hint_y",
        "racket_top_x", "racket_top_y",
        "racket_left_x", "racket_left_y",
        "racket_center_x", "racket_center_y",
        "racket_right_x", "racket_right_y",
        "racket_grip_end_x", "racket_grip_end_y",
        "racket_face_angle_deg",
        "racket_shaft_angle_deg",
        "racket_face_width_px",
        "racket_length_px",
        "racket_face_to_shaft_angle_deg",
        "racket_head_box_detected",
        "racket_detected",
    ]
    for key in keys:
        row[key] = None
    row["racket_detected"] = 0


def add_racket_fields(row, racket_result):
    if racket_result is None:
        add_empty_racket_fields(row)
        return

    for key, value in racket_result.items():
        row[key] = value
    row["racket_detected"] = 1


def add_derived_metrics(rows, fps):
    prev_left_wrist = None
    prev_right_wrist = None
    prev_hip_rel_x = None

    for row in rows:
        left_wrist = None
        right_wrist = None

        if row.get("left_wrist_base_right_heel_px_x") is not None and row.get("left_wrist_base_right_heel_px_y") is not None:
            left_wrist = (
                float(row["left_wrist_base_right_heel_px_x"]),
                float(row["left_wrist_base_right_heel_px_y"]),
            )

        if row.get("right_wrist_base_right_heel_px_x") is not None and row.get("right_wrist_base_right_heel_px_y") is not None:
            right_wrist = (
                float(row["right_wrist_base_right_heel_px_x"]),
                float(row["right_wrist_base_right_heel_px_y"]),
            )

        left_speed = None
        right_speed = None
        hip_speed = None

        if left_wrist is not None and prev_left_wrist is not None:
            left_speed = distance(left_wrist, prev_left_wrist) * fps

        if right_wrist is not None and prev_right_wrist is not None:
            right_speed = distance(right_wrist, prev_right_wrist) * fps

        hip_rel_x = row.get("hip_center_base_right_heel_px_x")
        if hip_rel_x is not None and prev_hip_rel_x is not None:
            hip_speed = abs(float(hip_rel_x) - float(prev_hip_rel_x)) * fps

        row["left_wrist_speed_px_per_sec"] = left_speed
        row["right_wrist_speed_px_per_sec"] = right_speed
        row["hip_speed_px_per_sec"] = hip_speed

        if left_speed is None and right_speed is None:
            row["hitting_side"] = None
            row["hit_wrist_speed_px_per_sec"] = None
            row["hit_wrist_px_x"] = None
            row["hit_wrist_px_y"] = None
            row["hit_wrist_base_right_heel_px_x"] = None
            row["hit_wrist_base_right_heel_px_y"] = None
        else:
            if right_speed is None:
                hit_side = "left"
            elif left_speed is None:
                hit_side = "right"
            else:
                hit_side = "right" if right_speed >= left_speed else "left"

            row["hitting_side"] = hit_side
            if hit_side == "right":
                row["hit_wrist_speed_px_per_sec"] = right_speed
                row["hit_wrist_px_x"] = row.get("right_wrist_px_x")
                row["hit_wrist_px_y"] = row.get("right_wrist_px_y")
                row["hit_wrist_base_right_heel_px_x"] = row.get("right_wrist_base_right_heel_px_x")
                row["hit_wrist_base_right_heel_px_y"] = row.get("right_wrist_base_right_heel_px_y")
            else:
                row["hit_wrist_speed_px_per_sec"] = left_speed
                row["hit_wrist_px_x"] = row.get("left_wrist_px_x")
                row["hit_wrist_px_y"] = row.get("left_wrist_px_y")
                row["hit_wrist_base_right_heel_px_x"] = row.get("left_wrist_base_right_heel_px_x")
                row["hit_wrist_base_right_heel_px_y"] = row.get("left_wrist_base_right_heel_px_y")

        prev_left_wrist = left_wrist
        prev_right_wrist = right_wrist
        prev_hip_rel_x = row.get("hip_center_base_right_heel_px_x")


def save_graphs(rows, graph_output_path):
    if not rows:
        return

    graph_columns = [
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
        "left_shoulder_angle_deg",
        "right_shoulder_angle_deg",
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "left_palm_angle_deg",
        "right_palm_angle_deg",
        "trunk_angle_deg",
        "shoulder_line_angle_deg",
        "hip_line_angle_deg",
        "gaze_angle_deg",
        "racket_face_angle_deg",
        "racket_face_to_shaft_angle_deg",
        "left_wrist_base_right_heel_px_x",
        "left_wrist_base_right_heel_px_y",
        "right_wrist_base_right_heel_px_x",
        "right_wrist_base_right_heel_px_y",
        "hip_center_base_right_heel_px_x",
        "hip_center_base_right_heel_px_y",
        "shoulder_center_base_right_heel_px_x",
        "shoulder_center_base_right_heel_px_y",
    ]

    time_values = [safe_float(row.get("time_sec")) for row in rows]

    valid_columns = []
    for column in graph_columns:
        y_values = [safe_float(row.get(column)) for row in rows]
        if not np.all(np.isnan(y_values)):
            valid_columns.append(column)

    if not valid_columns:
        return

    cols = 2
    plot_rows = math.ceil(len(valid_columns) / cols)

    fig, axes = plt.subplots(plot_rows, cols, figsize=(14, plot_rows * 3.5))

    if plot_rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    for i, column in enumerate(valid_columns):
        y_values = [safe_float(row.get(column)) for row in rows]
        axes[i].plot(time_values, y_values, color="blue")
        axes[i].set_title(column)
        axes[i].set_xlabel("Time (sec)")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)

    for j in range(len(valid_columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(graph_output_path, dpi=150)
    plt.close(fig)


def analyze_video(input_video_path: str, result_dir: Path, progress_bar=None, progress_text=None):
    cap = open_video_or_error(input_video_path, "動画を開けませんでした。")
    out = None
    csv_rows = []

    try:
        fps, width, height, frame_count = get_video_info(cap)
        if width <= 0 or height <= 0:
            raise RuntimeError("動画サイズを取得できませんでした。")
        if frame_count <= 0:
            raise RuntimeError("フレーム数を取得できませんでした。")

        output_video_path = result_dir / RESULT_VIDEO_NAME
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("解析結果動画の保存を開始できませんでした。")

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                annotated_frame = frame.copy()

                row = {
                    "frame": frame_index,
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                }

                left_wrist_px = None
                right_wrist_px = None
                left_elbow_px = None
                right_elbow_px = None

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    pose_row, left_wrist_px, right_wrist_px, left_elbow_px, right_elbow_px = extract_pose_row_data(
                        landmarks, width, height
                    )
                    row.update(pose_row)

                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                else:
                    add_empty_pose_fields(row)

                racket_result = detect_tennis_racket_five_points(
                    frame,
                    left_wrist_px,
                    right_wrist_px,
                    left_elbow_px,
                    right_elbow_px,
                )
                add_racket_fields(row, racket_result)

                draw_racket_five_points(annotated_frame, row)

                draw_text_unicode(
                    annotated_frame,
                    [
                        f"frame: {frame_index}",
                        f"time: {frame_index / fps:.3f} sec",
                    ],
                    20,
                    20,
                    font_size=26,
                    fill=(255, 255, 255),
                    stroke_fill=(0, 0, 0),
                    line_gap=6,
                )

                out.write(annotated_frame)
                csv_rows.append(row)
                frame_index += 1

                if frame_count > 0 and progress_bar is not None:
                    progress = min(frame_index / frame_count, 1.0)
                    progress_bar.progress(progress)
                    if progress_text is not None:
                        progress_text.text(f"解析中: {frame_index} / {frame_count} フレーム")

        add_derived_metrics(csv_rows, fps)

        output_csv_path = result_dir / RESULT_CSV_NAME
        all_fieldnames = set()
        for row in csv_rows:
            all_fieldnames.update(row.keys())

        fieldnames = ["frame", "frame_index", "time_sec"] + sorted(
            [name for name in all_fieldnames if name not in ("frame", "frame_index", "time_sec")]
        )

        with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        return {
            "fps": fps,
            "frame_count": frame_count,
            "rows": csv_rows,
            "video_path": output_video_path,
            "csv_path": output_csv_path,
        }

    finally:
        safe_release(cap)
        if out is not None:
            out.release()


def save_static_frames(input_video_path: str, static_dir: Path, rows, progress_bar=None, progress_text=None):
    cap = open_video_or_error(input_video_path, "動画を再オープンできませんでした。")

    try:
        selected_indices = build_even_indices(len(rows), STATIC_FRAME_COUNT)

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            total = len(selected_indices)
            for static_no, row_idx in enumerate(selected_indices, start=1):
                row = rows[row_idx]
                frame = read_frame_at(cap, row["frame_index"])
                if frame is None:
                    continue

                annotated_frame = frame.copy()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                draw_racket_five_points(annotated_frame, row)

                draw_text_unicode(
                    annotated_frame,
                    [
                        f"No: {static_no}",
                        f"frame_index: {row['frame_index']}",
                        f"time_sec: {row['time_sec']:.3f}",
                    ],
                    20,
                    20,
                    font_size=30,
                    fill=(255, 255, 255),
                    stroke_fill=(0, 0, 0),
                    line_gap=8,
                )

                file_name = f"{static_no:03d}_frame_{row['frame_index']}.png"
                save_ok = save_image_unicode(annotated_frame, static_dir / file_name)
                if not save_ok:
                    raise RuntimeError(f"静止画保存に失敗しました: {file_name}")

                if total > 0 and progress_bar is not None:
                    progress = min(static_no / total, 1.0)
                    progress_bar.progress(progress)
                    if progress_text is not None:
                        progress_text.text(f"静止画保存中: {static_no} / {total}")

    finally:
        safe_release(cap)


def zip_result_folder(result_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in result_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(result_dir.parent))


def build_result_folder_path(base_output_dir, input_video_name):
    raw_name = Path(input_video_name).stem
    safe_name = sanitize_windows_name(raw_name)
    return Path(base_output_dir) / f"result_{safe_name}"


def run_analysis(uploaded_file_bytes: bytes, uploaded_file_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        input_video_path = temp_root / uploaded_file_name
        input_video_path.write_bytes(uploaded_file_bytes)

        result_dir = build_result_folder_path(temp_root, uploaded_file_name)
        result_dir.mkdir(parents=True, exist_ok=True)
        static_dir = result_dir / STATIC_FRAMES_DIR_NAME
        static_dir.mkdir(parents=True, exist_ok=True)

        analyze_bar = st.progress(0.0)
        analyze_text = st.empty()
        analysis_result = analyze_video(
            input_video_path=str(input_video_path),
            result_dir=result_dir,
            progress_bar=analyze_bar,
            progress_text=analyze_text,
        )
        analyze_bar.empty()
        analyze_text.empty()

        static_bar = st.progress(0.0)
        static_text = st.empty()
        save_static_frames(
            input_video_path=str(input_video_path),
            static_dir=static_dir,
            rows=analysis_result["rows"],
            progress_bar=static_bar,
            progress_text=static_text,
        )
        static_bar.empty()
        static_text.empty()

        graph_path = result_dir / RESULT_GRAPH_NAME
        save_graphs(analysis_result["rows"], graph_path)

        zip_path = temp_root / f"{result_dir.name}.zip"
        zip_result_folder(result_dir, zip_path)

        return {
            "zip_bytes": zip_path.read_bytes(),
            "result_dir_name": result_dir.name,
            "output_video_bytes": (result_dir / RESULT_VIDEO_NAME).read_bytes(),
            "csv_bytes": (result_dir / RESULT_CSV_NAME).read_bytes(),
            "graph_bytes": graph_path.read_bytes() if graph_path.exists() else None,
        }


def main():
    st.set_page_config(page_title="動画解析アプリ", layout="wide")
    st.title("動画解析アプリ")
    st.write("MP4動画をアップロードして解析します。")

    uploaded_file = st.file_uploader("動画ファイルを選択してください", type=["mp4"])

    if uploaded_file is None:
        return

    st.video(uploaded_file)

    if st.button("解析開始"):
        try:
            with st.spinner("解析中です"):
                result = run_analysis(uploaded_file.getvalue(), uploaded_file.name)

            st.success("解析が完了しました。")

            st.download_button(
                label="結果一式をZIPでダウンロード",
                data=result["zip_bytes"],
                file_name=f"{result['result_dir_name']}.zip",
                mime="application/zip",
            )

            st.download_button(
                label="analysis.csv をダウンロード",
                data=result["csv_bytes"],
                file_name="analysis.csv",
                mime="text/csv",
            )

            if result["graph_bytes"] is not None:
                st.image(result["graph_bytes"], caption="graphs.png")

            st.video(result["output_video_bytes"])

        except Exception:
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
