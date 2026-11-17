import cv2
import numpy as np
import csv
import math
from scipy.signal import find_peaks

VIDEO_PATH = "30度旋转鸟瞰图.mp4"

OUTPUT_VIDEO = "30度旋转检测.mp4"
OUTPUT_CSV = "30度旋转检测.csv"

ROI_Y_RATIO = 0.45       # 使用画面下方 55%
MIN_LINE_LENGTH = 30.0

ANGLE_BINS = 180
HIST_SMOOTH_SIGMA = 2.0

MAX_PEAKS = 8

ORTHOGONAL_TARGET = 90.0
ORTHOGONAL_TOL = 35.0

MAX_FRAME_JUMP = 35.0

DISPLAY_SCALE = 0.65

def normalize180(angle):
    return angle % 180.0

def angle_diff(a, b):
    return (b - a + 90.0) % 180.0 - 90.0

def circular_mean_180(angles, weights=None):
    angles = np.asarray(angles, dtype=np.float64)

    if len(angles) == 0:
        return None

    if weights is None:
        weights = np.ones_like(angles)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    theta = np.deg2rad(angles * 2.0)

    s = np.sum(weights * np.sin(theta))
    c = np.sum(weights * np.cos(theta))

    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return normalize180(float(angles[0]))

    result = np.rad2deg(np.arctan2(s, c)) / 2.0
    return normalize180(result)

def gaussian_smooth_hist(hist, sigma=2.0):
    radius = int(max(2, sigma * 3))
    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    padded = np.concatenate([
        hist[-radius:],
        hist,
        hist[:radius]
    ])

    smoothed = np.convolve(padded, kernel, mode="same")

    return smoothed[radius:radius + len(hist)]

def build_lsd():
    return cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

def detect_lines(gray, lsd):
    detected = lsd.detect(gray)[0]

    if detected is None:
        return []

    result = []

    for item in detected:
        x1, y1, x2, y2 = item[0]

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < MIN_LINE_LENGTH:
            continue

        angle = normalize180(math.degrees(math.atan2(dy, dx)))

        result.append(
            (x1, y1, x2, y2, length, angle)
        )

    return result

def build_orientation_histogram(lines):
    hist = np.zeros(ANGLE_BINS, dtype=np.float64)

    for _, _, _, _, length, angle in lines:
        idx = int(angle) % ANGLE_BINS
        hist[idx] += length

    return hist

def refine_peak_direction(lines, peak_angle, radius=5):
    selected_angles = []
    selected_weights = []

    for *_, length, angle in lines:
        d = abs(angle_diff(peak_angle, angle))

        if d <= radius:
            selected_angles.append(angle)
            selected_weights.append(length)

    if not selected_angles:
        return normalize180(peak_angle)

    return circular_mean_180(
        selected_angles,
        selected_weights
    )

def find_direction_candidates(lines):
    if not lines:
        return []

    hist = build_orientation_histogram(lines)

    if np.max(hist) <= 0:
        return []

    smooth = gaussian_smooth_hist(
        hist,
        HIST_SMOOTH_SIGMA
    )

    extended = np.concatenate([
        smooth,
        smooth,
        smooth
    ])

    peaks, properties = find_peaks(
        extended,
        distance=10,
        prominence=max(np.max(smooth) * 0.03, 1.0)
    )

    candidates = []

    for p in peaks:
        if ANGLE_BINS <= p < 2 * ANGLE_BINS:
            angle = p - ANGLE_BINS
            score = smooth[int(angle)]

            refined = refine_peak_direction(
                lines,
                angle,
                radius=5
            )

            candidates.append({
                "angle": normalize180(refined),
                "score": float(score)
            })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    unique = []

    for c in candidates:
        if all(
            abs(angle_diff(c["angle"], u["angle"])) > 8
            for u in unique
        ):
            unique.append(c)

        if len(unique) >= MAX_PEAKS:
            break

    return unique

def choose_two_families(candidates, prev_a, prev_b):

    if not candidates:
        return None, None

    if prev_a is None or prev_b is None:

        a = candidates[0]

        b_candidates = [
            c for c in candidates[1:]
            if abs(
                abs(angle_diff(a["angle"], c["angle"]))
                - ORTHOGONAL_TARGET
            ) <= ORTHOGONAL_TOL
        ]

        if not b_candidates:
            return a, None

        b = min(
            b_candidates,
            key=lambda c:
            abs(
                abs(angle_diff(a["angle"], c["angle"]))
                - ORTHOGONAL_TARGET
            )
        )

        return a, b

    def nearest_to(prev_angle):
        valid = [
            c for c in candidates
            if abs(angle_diff(prev_angle, c["angle"]))
            <= MAX_FRAME_JUMP
        ]

        if not valid:
            return None

        return min(
            valid,
            key=lambda c:
            abs(angle_diff(prev_angle, c["angle"]))
        )

    a = nearest_to(prev_a)
    b = nearest_to(prev_b)

    if a is not None and b is not None:
        if abs(a["angle"] - b["angle"]) < 1e-6:
            da = abs(angle_diff(prev_a, a["angle"]))
            db = abs(angle_diff(prev_b, b["angle"]))

            if da <= db:
                b = None
            else:
                a = None

    return a, b

def draw_family_lines(
    frame,
    lines,
    family_a,
    family_b
):

    canvas = frame.copy()

    for x1, y1, x2, y2, length, angle in lines:

        color = (130, 130, 130)
        thickness = 1

        if family_a is not None:
            da = abs(angle_diff(family_a, angle))

            if da <= 8:
                color = (0, 255, 0)
                thickness = 2

        if family_b is not None:
            db = abs(angle_diff(family_b, angle))

            if db <= 8:
                color = (255, 0, 0)
                thickness = 2

        cv2.line(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )

    return canvas

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("=" * 70)
    print("双方向光伏板线方向跟踪")
    print("=" * 70)
    print(f"视频：{VIDEO_PATH}")
    print(f"FPS：{fps:.3f}")
    print(f"总帧数：{total_frames}")
    print(
        f"视频时长："
        f"{total_frames / fps:.2f}s"
    )
    print(
        f"ROI：y >= "
        f"{int(height * ROI_Y_RATIO)}"
    )
    print("=" * 70)

    out_width = int(width * DISPLAY_SCALE)
    out_height = int(height * DISPLAY_SCALE)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height)
    )

    lsd = build_lsd()

    prev_a = None
    prev_b = None

    cumulative_a = 0.0
    cumulative_b = 0.0

    csv_rows = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        time_sec = (
            (frame_idx - 1) / fps
        )

        roi_y = int(
            height * ROI_Y_RATIO
        )

        roi = frame[
            roi_y:height,
            0:width
        ]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        roi_lines = detect_lines(
            gray,
            lsd
        )

        lines = []

        for x1, y1, x2, y2, length, angle in roi_lines:

            lines.append(
                (
                    x1,
                    y1 + roi_y,
                    x2,
                    y2 + roi_y,
                    length,
                    angle
                )
            )

        candidates = find_direction_candidates(
            lines
        )

        a_candidate, b_candidate = \
            choose_two_families(
                candidates,
                prev_a,
                prev_b
            )

        current_a = (
            a_candidate["angle"]
            if a_candidate is not None
            else None
        )

        current_b = (
            b_candidate["angle"]
            if b_candidate is not None
            else None
        )

        delta_a = None
        delta_b = None

        if (
            current_a is not None
            and prev_a is not None
        ):
            delta_a = angle_diff(
                prev_a,
                current_a
            )
            cumulative_a += delta_a

        if (
            current_b is not None
            and prev_b is not None
        ):
            delta_b = angle_diff(
                prev_b,
                current_b
            )
            cumulative_b += delta_b

        if current_a is not None:
            prev_a = current_a

        if current_b is not None:
            prev_b = current_b

        orthogonal_diff = None

        if (
            current_a is not None
            and current_b is not None
        ):
            orthogonal_diff = abs(
                angle_diff(
                    current_a,
                    current_b
                )
            )


        vis = draw_family_lines(
            frame,
            lines,
            current_a,
            current_b
        )

        cv2.line(
            vis,
            (0, roi_y),
            (width, roi_y),
            (0, 255, 255),
            2
        )

        panel_h = 180

        overlay = vis.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, panel_h),
            (20, 20, 20),
            -1
        )

        vis = cv2.addWeighted(
            overlay,
            0.65,
            vis,
            0.35,
            0
        )

        cv2.putText(
            vis,
            f"Frame: {frame_idx}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            f"Time: {time_sec:.3f}s",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        if current_a is not None:
            cv2.putText(
                vis,
                f"A direction: {current_a:.2f} deg",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                vis,
                "A direction: None",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2
            )

        if current_b is not None:
            cv2.putText(
                vis,
                f"B direction: {current_b:.2f} deg",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 0, 0),
                2
            )
        else:
            cv2.putText(
                vis,
                "B direction: None",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 0, 0),
                2
            )

        if orthogonal_diff is not None:
            cv2.putText(
                vis,
                f"A-B: {orthogonal_diff:.2f} deg",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 0),
                2
            )

        cv2.putText(
            vis,
            "GREEN = family A    BLUE = family B    GRAY = other LSD lines",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        vis_small = cv2.resize(
            vis,
            (out_width, out_height),
            interpolation=cv2.INTER_AREA
        )

        writer.write(vis_small)

        csv_rows.append([
            frame_idx,
            time_sec,
            current_a,
            current_b,
            delta_a,
            delta_b,
            cumulative_a,
            cumulative_b,
            orthogonal_diff,
            len(lines),
            len(candidates)
        ])

    cap.release()
    writer.release()

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer_csv = csv.writer(f)

        writer_csv.writerow([
            "frame",
            "time_sec",
            "direction_A_deg",
            "direction_B_deg",
            "delta_A_deg",
            "delta_B_deg",
            "cumulative_A_deg",
            "cumulative_B_deg",
            "abs_A_B_deg",
            "num_lines",
            "num_candidates"
        ])

        writer_csv.writerows(csv_rows)

    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)
    print(
        f"双方向可视化视频："
        f"{OUTPUT_VIDEO}"
    )
    print(
        f"CSV 日志："
        f"{OUTPUT_CSV}"
    )
    print()
    print(
        "最终累计 A："
        f"{cumulative_a:.2f}°"
    )
    print(
        "最终累计 B："
        f"{cumulative_b:.2f}°"
    )
    # 计算两组方向累计旋转量的平均值
    mean_rotation = (
                            cumulative_a + cumulative_b
                    ) / 2.0

    # 计算两组结果之间的差异
    rotation_difference = abs(
        cumulative_a - cumulative_b
    )

    print(
        "平均累计旋转："
        f"{mean_rotation:.2f}°"
    )

    print(
        "A/B结果差异："
        f"{rotation_difference:.2f}°"
    )

    print("=" * 70)

if __name__ == "__main__":
    main()
