# -*- coding: utf-8 -*-
"""LSD-based arbitrary-angle rotation estimation on solar-panel video.

The panel lines are treated as a static world reference. Their image direction is
tracked with a 180-degree-periodic angle. Frame-to-frame minimal differences are
unwrapped and accumulated, so cumulative rotation is not limited to 180 degrees.
"""
import cv2
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt

VIDEO_PATH = "180度旋转鸟瞰图.mp4"
PROCESS_WIDTH = 1000
ROI_Y_RATIO = 0.45
MIN_LINE_LENGTH = 30.0
ANGLE_BIN_SIZE = 1.0
ANGLE_SMOOTH_WINDOW = 5
MIN_VALID_LINES = 5
MAX_FRAME_ANGLE_CHANGE = 25.0
SHOW_REALTIME = True
SAVE_VIDEO = True
OUTPUT_CSV = "solar_panel_rotation_result.csv"
OUTPUT_PLOT = "solar_panel_rotation_curve.png"
OUTPUT_VIDEO = "solar_panel_rotation_annotated.mp4"


def normalize_line_angle(angle):
    return angle % 180.0


def line_angle(x1, y1, x2, y2):
    return normalize_line_angle(np.degrees(np.arctan2(y2-y1, x2-x1)))


def angle_diff_180(a, b):
    """Minimal signed difference for undirected line angles, range [-90,90)."""
    return (b - a + 90.0) % 180.0 - 90.0


def circular_mean_180(angles, weights):
    if not angles:
        return None
    a = np.asarray(angles, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.deg2rad(2.0*a)
    x, y = np.sum(w*np.cos(r)), np.sum(w*np.sin(r))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return None
    return (0.5*np.degrees(np.arctan2(y, x))) % 180.0


def detect_panel_direction(gray, lsd, previous_angle=None):
    h, _ = gray.shape
    y0 = int(h*ROI_Y_RATIO)
    roi = cv2.GaussianBlur(gray[y0:, :], (3, 3), 0)
    detected = lsd.detect(roi)[0]
    if detected is None:
        return None, [], 0.0

    items = []
    for line in detected:
        x1, y1, x2, y2 = map(float, line[0])
        y1 += y0; y2 += y0
        length = float(np.hypot(x2-x1, y2-y1))
        if length < MIN_LINE_LENGTH:
            continue
        items.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "angle": line_angle(x1, y1, x2, y2), "length": length
        })

    if len(items) < MIN_VALID_LINES:
        return None, items, 0.0

    hist = np.zeros(180, dtype=float)
    for it in items:
        hist[min(int(it["angle"]), 179)] += it["length"]
    kernel = np.ones(5, dtype=float)/5.0
    hs = np.convolve(np.r_[hist[-2:], hist, hist[:2]], kernel, mode="valid")

    peaks = []
    for i in range(180):
        if hs[i] >= hs[(i-1)%180] and hs[i] >= hs[(i+1)%180]:
            peaks.append(i)
    peaks.sort(key=lambda i: hs[i], reverse=True)
    peaks = peaks[:8]
    if not peaks:
        return None, items, 0.0

    candidates = [p+0.5 for p in peaks]
    weights = [hs[p] for p in peaks]

    if previous_angle is not None:
        valid = []
        for a, w in zip(candidates, weights):
            d = abs(angle_diff_180(previous_angle, a))
            if d <= MAX_FRAME_ANGLE_CHANGE:
                valid.append((w/(1.0+0.15*d), a, w))
        if valid:
            _, selected, selected_weight = max(valid)
        else:
            p = int(np.argmax(hs)); selected, selected_weight = p+0.5, hs[p]
    else:
        p = int(np.argmax(hs)); selected, selected_weight = p+0.5, hs[p]

    near_a, near_w = [], []
    for it in items:
        if abs(angle_diff_180(selected, it["angle"])) <= 12.0:
            near_a.append(it["angle"]); near_w.append(it["length"])
    refined = circular_mean_180(near_a, near_w)
    if refined is not None:
        selected = refined

    total = sum(it["length"] for it in items)
    confidence = float(selected_weight/max(total, 1e-6))
    return selected, items, confidence


def main():
    vp = Path(VIDEO_PATH)
    if not vp.exists():
        raise FileNotFoundError(f"找不到视频：{vp.resolve()}；请修改 VIDEO_PATH")
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{vp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, PROCESS_WIDTH/max(ow,1))
    w, h = int(ow*scale), int(oh*scale)

    print("="*68)
    print("Solar Panel Arbitrary-Angle Rotation Estimation")
    print(f"FPS={fps:.3f}, frames={nframes}, duration={nframes/fps:.2f}s")
    print(f"process size={w}x{h}")
    print("LSD line angle: 0~180 degree periodic; cumulative rotation is unwrapped frame-by-frame.")
    print("="*68)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w,h))

    times=[]; dirs=[]; cum=[]; confs=[]; valid=[]
    history=[]; prev=None; cumulative=0.0; i=0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if scale != 1.0:
            frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        direction, lines, confidence = detect_panel_direction(gray,lsd,prev)
        ok=direction is not None

        if ok:
            history.append(direction)
            if len(history)>ANGLE_SMOOTH_WINDOW: history.pop(0)
            smoothed=circular_mean_180(history,np.ones(len(history)))
            if prev is not None:
                cumulative += angle_diff_180(prev,smoothed)
            prev=smoothed
        else:
            smoothed=prev

        t=i/fps
        times.append(t); dirs.append(np.nan if smoothed is None else smoothed)
        cum.append(cumulative); confs.append(confidence); valid.append(ok)

        display=frame.copy(); y0=int(h*ROI_Y_RATIO)
        cv2.line(display,(0,y0),(w,y0),(255,255,0),2)
        for it in lines:
            cv2.line(display,(it["x1"],it["y1"]),(it["x2"],it["y2"]),(0,255,0),1)
        cv2.rectangle(display,(12,12),(720,125),(0,0,0),-1)
        status="PANEL DIRECTION TRACKING" if ok else "NO RELIABLE PANEL DIRECTION"
        cv2.putText(display,status,(28,42),cv2.FONT_HERSHEY_SIMPLEX,0.72,(0,255,255),2)
        cv2.putText(display,f"Panel direction: {smoothed:.2f} deg" if smoothed is not None else "Panel direction: ---",(28,72),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
        cv2.putText(display,f"Cumulative rotation: {cumulative:.2f} deg",(28,102),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,255,0),2)
        cv2.putText(display,f"Time: {t:.2f}s",(w-170,35),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
        if out is not None: out.write(display)
        if SHOW_REALTIME:
            cv2.imshow("Solar Panel Rotation Estimation",display)
            key=cv2.waitKey(1)&0xff
            if key in (27,ord('q')): break
        i+=1

    cap.release()
    if out is not None: out.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8-sig") as f:
        cw=csv.writer(f); cw.writerow(["frame","time_s","panel_direction_deg","cumulative_rotation_deg","confidence","valid"])
        for j in range(len(times)):
            cw.writerow([j,f"{times[j]:.6f}","" if np.isnan(dirs[j]) else f"{dirs[j]:.6f}",f"{cum[j]:.6f}",f"{confs[j]:.6f}",int(valid[j])])

    fig=plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(2,1,1); ax1.plot(times,dirs,label="Tracked panel direction"); ax1.set_ylim(-5,185); ax1.set_title("Tracked Solar Panel Direction"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Direction (deg, 0~180)"); ax1.grid(True,alpha=.3); ax1.legend()
    ax2=fig.add_subplot(2,1,2); ax2.plot(times,cum,label="Cumulative visual rotation"); ax2.axhline(0,linestyle="--",linewidth=1); ax2.set_title("Cumulative Rotation (can exceed 180° / 360°)"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Rotation (deg)"); ax2.grid(True,alpha=.3); ax2.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_PLOT,dpi=160,bbox_inches="tight"); plt.close(fig)

    if cum:
        print("\n最终结果")
        vd=[x for x in dirs if not np.isnan(x)]
        if vd: print(f"初始方向：{vd[0]:.2f}°  最终方向：{vd[-1]:.2f}°")
        print(f"累计旋转：{cum[-1]:.2f}°")
        print(f"累计旋转绝对值：{abs(cum[-1]):.2f}°")
        print(f"CSV: {OUTPUT_CSV}")
        print(f"曲线: {OUTPUT_PLOT}")
        if SAVE_VIDEO: print(f"视频: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
