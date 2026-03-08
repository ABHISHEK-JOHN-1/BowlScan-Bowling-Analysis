"""
BowlScan — Bowling Biomechanics Analysis Engine
Full engine with all 8 metrics, 3D pose tracking, PDF report builder.
Import and call run_analysis() from the Streamlit app.
"""

import os
import csv
import time
import base64
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ---- Thresholds -----------------------------------------------------------------

HSS_MIXED_THRESHOLD       = 20.0
SCR_MIXED_THRESHOLD       = 30.0
TRUNK_LATERAL_RISK        = 45.0
FRONT_KNEE_RISK           = 13.0
EXT_RATE_RISK             = 8.0
MIN_INTERIOR_ANGLE_RISK   = 120.0
ACTION_SIDE_ON_THRESHOLD  = 60.0
ACTION_FRONT_ON_THRESHOLD = 30.0


# ---- Signal processing ----------------------------------------------------------

def butter_lowpass_filter(data, cutoff=3.0, fs=30.0, order=4):
    data = np.array(data, dtype=float)
    if len(data) <= 15:
        return data
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# ---- Geometry helpers -----------------------------------------------------------

def angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def hip_shoulder_separation(hip_l, hip_r, sh_l, sh_r):
    hip_vec = np.array([hip_r[0] - hip_l[0], hip_r[2] - hip_l[2]])
    sh_vec  = np.array([sh_r[0]  - sh_l[0],  sh_r[2]  - sh_l[2]])
    cos_val = np.dot(hip_vec, sh_vec) / (np.linalg.norm(hip_vec) * np.linalg.norm(sh_vec) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def shoulder_alignment_angle(sh_l, sh_r):
    sh_vec   = np.array([sh_r[0] - sh_l[0], sh_r[2] - sh_l[2]])
    bowl_dir = np.array([0.0, 1.0])
    cos_val  = np.dot(sh_vec, bowl_dir) / (np.linalg.norm(sh_vec) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(cos_val), 0.0, 1.0))))


def classify_action(sh_angle, hss_angle):
    is_mixed = hss_angle >= HSS_MIXED_THRESHOLD
    if sh_angle >= ACTION_SIDE_ON_THRESHOLD:
        base = "Side-on"
    elif sh_angle < ACTION_FRONT_ON_THRESHOLD:
        base = "Front-on"
    else:
        base = "Semi-open"
    if is_mixed:
        return f"Mixed ({base} base)", True
    return base, False


def trunk_lateral_flexion_angle(sh_l_2d, sh_r_2d, hip_l_2d, hip_r_2d):
    sh_mid  = ((sh_l_2d[0] + sh_r_2d[0]) / 2, (sh_l_2d[1] + sh_r_2d[1]) / 2)
    hip_mid = ((hip_l_2d[0] + hip_r_2d[0]) / 2, (hip_l_2d[1] + hip_r_2d[1]) / 2)
    dx = sh_mid[0] - hip_mid[0]
    dy = sh_mid[1] - hip_mid[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))


# ---- PDF report builder ---------------------------------------------------------

def build_pdf_report(pdf_path, bowler_name, graph_path, data):
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    bg_color  = colors.HexColor('#0d0d1a')
    accent    = colors.HexColor('#00c8ff')
    green_c   = colors.HexColor('#00ff88')
    red_c     = colors.HexColor('#ff4444')
    orange_c  = colors.HexColor('#ffaa00')
    white_c   = colors.HexColor('#e8e8ff')
    dim_c     = colors.HexColor('#888aaa')
    panel_c   = colors.HexColor('#111128')

    title_style   = ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18,
                                   textColor=accent, alignment=TA_CENTER, spaceAfter=4)
    sub_style     = ParagraphStyle('sub',   fontName='Helvetica',      fontSize=10,
                                   textColor=dim_c,  alignment=TA_CENTER, spaceAfter=12)
    section_style = ParagraphStyle('sec',  fontName='Helvetica-Bold', fontSize=11,
                                   textColor=accent, spaceAfter=6, spaceBefore=14)
    body_style    = ParagraphStyle('body', fontName='Helvetica',      fontSize=9,
                                   textColor=white_c, spaceAfter=4)

    icc_legal  = data['icc_legal']
    risk_level = data['risk_level']
    risk_color = green_c if risk_level == 'LOW' else (orange_c if risk_level == 'MODERATE' else red_c)
    icc_color  = green_c if icc_legal else red_c

    story = []
    story.append(Paragraph("BOWLING BIOMECHANICAL REPORT", title_style))
    story.append(Paragraph(
        f"Bowler: {bowler_name}    |    Generated: {time.strftime('%d %b %Y, %H:%M')}", sub_style
    ))

    verdict_text  = "LEGAL" if icc_legal else "ILLEGAL"
    verdict_color = green_c if icc_legal else red_c

    summary_data = [
        ['ICC VERDICT', 'BOWLING ACTION', 'INJURY RISK'],
        [
            Paragraph(
                f'<font color="{verdict_color.hexval()}">{verdict_text}</font>'
                f'<br/><font size="8" color="#888aaa">{data["icc_extension"]:.1f} deg extension</font>',
                ParagraphStyle('sc', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER)
            ),
            Paragraph(
                f'<font color="#e8e8ff">{data["action_type"]}</font>',
                ParagraphStyle('sc2', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER)
            ),
            Paragraph(
                f'<font color="{risk_color.hexval()}">{risk_level}</font>'
                f'<br/><font size="8" color="#888aaa">{data["score"]}/{data["total_flags"]} flags</font>',
                ParagraphStyle('sc3', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER)
            ),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[5.5*cm, 6.5*cm, 5.5*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a3a')),
        ('BACKGROUND',    (0,1), (-1,1), panel_c),
        ('TEXTCOLOR',     (0,0), (-1,0), accent),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,0), 9),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#2a2a4a')),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Analysis Charts", section_style))
    img = RLImage(graph_path, width=17*cm, height=14*cm)
    story.append(img)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Detailed Metrics", section_style))

    def status_text(ok):
        return "OK" if ok else "FLAGGED"

    metrics_rows = [
        ['Metric', 'Value', 'Threshold', 'Status'],
        ['ICC Elbow Extension',          f"{data['icc_extension']:.1f} deg",   '15.0 deg max',                           'LEGAL' if icc_legal else 'ILLEGAL'],
        ['Bowling Action',               data['action_type'],                   'Side/Front/Semi/Mixed',                   'Mixed' if data['is_mixed'] else 'OK'],
        ['Hip-Shoulder Sep. at Release', f"{data['hss_at_release']:.1f} deg",  f"below {HSS_MIXED_THRESHOLD} deg",        status_text(data['hss_at_release'] < HSS_MIXED_THRESHOLD)],
        ['Max Shoulder Counter-Rot.',    f"{data['scr_max']:.1f} deg",         f"below {SCR_MIXED_THRESHOLD} deg",        status_text(data['scr_max'] < SCR_MIXED_THRESHOLD)],
        ['Peak Trunk Lateral Flex.',     f"{data['peak_tlf']:.1f} deg",        f"below {TRUNK_LATERAL_RISK} deg",         status_text(data['peak_tlf'] <= TRUNK_LATERAL_RISK)],
        ['Front Knee Flexion at FFC',    f"{data['knee_at_release']:.1f} deg", f"above {FRONT_KNEE_RISK} deg",            status_text(data['knee_at_release'] >= FRONT_KNEE_RISK)],
        ['Min Interior Elbow Angle',     f"{data['min_interior']:.1f} deg",    f"above {MIN_INTERIOR_ANGLE_RISK} deg",    status_text(data['min_interior'] >= MIN_INTERIOR_ANGLE_RISK)],
        ['Elbow Extension Rate',         f"{data['ext_rate']:.1f} deg/frame",  f"below {EXT_RATE_RISK} deg/frame",        status_text(data['ext_rate'] <= EXT_RATE_RISK)],
    ]

    def row_bg(status):
        return colors.HexColor('#2a0a0a') if status in ('ILLEGAL', 'FLAGGED', 'Mixed') else panel_c

    metrics_table = Table(metrics_rows, colWidths=[6*cm, 3.5*cm, 4*cm, 4*cm])
    ts = [
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a3a')),
        ('TEXTCOLOR',     (0,0), (-1,0), accent),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#2a2a4a')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TEXTCOLOR',     (0,1), (0,-1), white_c),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
    ]
    for row_idx, row in enumerate(metrics_rows[1:], start=1):
        status = row[3]
        ts.append(('BACKGROUND', (0, row_idx), (-1, row_idx), row_bg(status)))
        clr = red_c if status in ('ILLEGAL', 'FLAGGED', 'Mixed') else green_c
        ts.append(('TEXTCOLOR', (3, row_idx), (3, row_idx), clr))
        ts.append(('FONTNAME',  (3, row_idx), (3, row_idx), 'Helvetica-Bold'))
        ts.append(('TEXTCOLOR', (1, row_idx), (2, row_idx), white_c))
    metrics_table.setStyle(TableStyle(ts))
    story.append(metrics_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Injury Risk Checklist", section_style))
    for label, triggered in data['risk_flags'].items():
        color_hex = '#ff4444' if triggered else '#00ff88'
        tag       = 'FLAGGED' if triggered else 'OK'
        story.append(Paragraph(
            f'<font color="{color_hex}"><b>[{tag}]</b></font>  <font color="#e8e8ff">{label}</font>',
            body_style
        ))

    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("References", section_style))
    refs = [
        "Portus et al. (2004) — Hip and shoulder separation in fast bowling",
        "Elliott et al. (1992) — Shoulder counter-rotation and lumbar stress fractures",
        "Crewe et al. — Front knee angle and lumbar spine injury risk",
        "Quintic / ECB Coaching Manual — Trunk lateral flexion thresholds",
        "PMC 10519895 — Front knee mechanics in cricket fast bowling",
        "Aginsky & Noakes (2010) — ICC elbow extension measurement methodology",
    ]
    for ref in refs:
        story.append(Paragraph(
            f"• {ref}",
            ParagraphStyle('ref', fontName='Helvetica', fontSize=7, textColor=dim_c, spaceAfter=2)
        ))

    doc.build(story)


# ---- Main analysis function ------------------------------------------------------

def run_analysis(video_path, bowler_arm="RIGHT", bowler_name="Unknown Bowler",
                 start_frame=0, end_frame=999999,
                 progress_callback=None, log_callback=None):
    """
    Run the full biomechanical analysis.

    Args:
        video_path      : path to video file
        bowler_arm      : "RIGHT" or "LEFT"
        bowler_name     : display name for reports
        start_frame     : first frame to process (trim)
        end_frame       : last frame to process (trim)
        progress_callback(pct: int)  : called with 0-100 during processing
        log_callback(msg: str)       : called with log messages

    Returns dict with all results, or raises on failure.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)

    def progress(pct):
        if progress_callback:
            progress_callback(int(pct))

    log("Opening video file...")

    mp_pose           = mp.solutions.pose
    pose              = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if bowler_arm == "RIGHT":
        S_IDX, E_IDX, W_IDX = 12, 14, 16
        KNEE_FRONT_IDX, HIP_FRONT_IDX, ANKLE_FRONT_IDX = 25, 23, 27
    else:
        S_IDX, E_IDX, W_IDX = 11, 13, 15
        KNEE_FRONT_IDX, HIP_FRONT_IDX, ANKLE_FRONT_IDX = 26, 24, 28

    HIP_L_IDX, HIP_R_IDX = 23, 24
    SH_L_IDX,  SH_R_IDX  = 11, 12

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracked_data       = []
    frame_count        = 0
    bfc_shoulder_angle = None
    bfc_hss_angle      = None
    bfc_captured       = False
    scr_max            = 0.0

    preview_frames = []
    preview_every  = max(1, total // 80)

    log(f"Video: {total} frames at {fps:.0f} fps")
    log("Running pose detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count < start_frame:
            continue
        if frame_count > end_frame:
            break

        progress(int((frame_count / max(total, 1)) * 60))

        frame_resized = cv2.resize(frame, (800, 600))
        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_world_landmarks and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            lands_2d = results.pose_landmarks.landmark
            w_lands  = results.pose_world_landmarks.landmark
            h, w_dim, _ = image_bgr.shape

            def px(idx):
                return (int(lands_2d[idx].x * w_dim), int(lands_2d[idx].y * h))

            def wld(idx):
                return [w_lands[idx].x, w_lands[idx].y, w_lands[idx].z]

            cv2.line(image_bgr, px(S_IDX), px(E_IDX), (0, 255, 255), 5)
            cv2.line(image_bgr, px(E_IDX), px(W_IDX), (0, 255, 255), 5)
            cv2.circle(image_bgr, px(E_IDX), 10, (0, 0, 255), -1)

            elbow_interior     = angle_3d(wld(S_IDX), wld(E_IDX), wld(W_IDX))
            hss                = hip_shoulder_separation(wld(HIP_L_IDX), wld(HIP_R_IDX), wld(SH_L_IDX), wld(SH_R_IDX))
            sh_align           = shoulder_alignment_angle(wld(SH_L_IDX), wld(SH_R_IDX))
            tlf                = trunk_lateral_flexion_angle(px(SH_L_IDX), px(SH_R_IDX), px(HIP_L_IDX), px(HIP_R_IDX))
            front_knee         = angle_3d(wld(HIP_FRONT_IDX), wld(KNEE_FRONT_IDX), wld(ANKLE_FRONT_IDX))
            front_knee_flexion = 180.0 - front_knee

            MIN_VIS            = 0.55
            wrist_vis          = lands_2d[W_IDX].visibility
            elbow_vis          = lands_2d[E_IDX].visibility
            shoulder_vis       = lands_2d[S_IDX].visibility
            landmarks_reliable = wrist_vis >= MIN_VIS and elbow_vis >= MIN_VIS and shoulder_vis >= MIN_VIS
            wrist_above_3d     = w_lands[W_IDX].y < w_lands[S_IDX].y
            arm_loading        = elbow_interior > 90.0 and wrist_vis >= 0.4
            arm_raised         = landmarks_reliable and (wrist_above_3d or arm_loading)

            if arm_raised and not bfc_captured:
                bfc_shoulder_angle = sh_align
                bfc_hss_angle      = hss
                bfc_captured       = True

            if bfc_captured and bfc_shoulder_angle is not None:
                scr_max = max(scr_max, abs(sh_align - bfc_shoulder_angle))

            if arm_raised:
                tracked_data.append((frame_count, elbow_interior, hss, sh_align, tlf, front_knee_flexion))
                label, label_color = "TRACKING", (0, 255, 0)
            else:
                label, label_color = "SCANNING", (0, 165, 255)

            cv2.putText(image_bgr, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
            cv2.putText(image_bgr, f"Elbow: {elbow_interior:.0f} deg", (30, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(image_bgr, f"HSS:   {hss:.0f} deg",            (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        if frame_count % preview_every == 0:
            _, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
            preview_frames.append(base64.b64encode(buf).decode('utf-8'))

    cap.release()

    if not tracked_data:
        raise ValueError("No delivery stride detected. Ensure the video shows a clear side-on bowling action.")

    log(f"Pose detection complete. {len(tracked_data)} frames tracked.")
    log("Isolating delivery stride...")
    progress(65)

    GAP_TOLERANCE = 20
    chunks, current_chunk = [], []
    for i in range(len(tracked_data)):
        if i > 0 and tracked_data[i][0] - tracked_data[i-1][0] > GAP_TOLERANCE:
            if len(current_chunk) > 5:
                chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append(tracked_data[i])
    if current_chunk and len(current_chunk) > 5:
        chunks.append(current_chunk)

    if not chunks:
        raise ValueError("Could not isolate a delivery chunk. Try trimming the video to a single delivery.")

    delivery_chunk = max(chunks, key=len)
    frames         = np.array([d[0] for d in delivery_chunk])
    raw_angles     = np.array([d[1] for d in delivery_chunk])
    raw_hss        = np.array([d[2] for d in delivery_chunk])
    raw_sh_align   = np.array([d[3] for d in delivery_chunk])
    raw_tlf        = np.array([d[4] for d in delivery_chunk])
    raw_knee_flex  = np.array([d[5] for d in delivery_chunk])

    log("Smoothing signals...")
    progress(70)

    sm_angles    = butter_lowpass_filter(raw_angles,    fs=fps)
    sm_hss       = butter_lowpass_filter(raw_hss,       fs=fps)
    sm_sh_align  = butter_lowpass_filter(raw_sh_align,  fs=fps)
    sm_tlf       = butter_lowpass_filter(raw_tlf,       fs=fps)
    sm_knee_flex = butter_lowpass_filter(raw_knee_flex, fs=fps)
    flexion      = 180.0 - sm_angles

    search_start  = len(sm_angles) // 3
    rel_local_idx = int(np.argmax(sm_angles[search_start:]))
    release_idx   = search_start + rel_local_idx

    PRE_WINDOW  = min(6, release_idx)
    POST_WINDOW = min(6, len(flexion) - release_idx - 1)
    pre_flex    = flexion[release_idx - PRE_WINDOW  : release_idx + 1]
    post_flex   = flexion[release_idx               : release_idx + POST_WINDOW + 1]

    max_flex_before = float(np.max(pre_flex))  if len(pre_flex)  else 0.0
    min_flex_after  = float(np.min(post_flex)) if len(post_flex) else 0.0
    icc_extension   = max(0.0, max_flex_before - min_flex_after)
    icc_legal       = icc_extension <= 15.0

    bfc_sh_used  = bfc_shoulder_angle if bfc_shoulder_angle is not None else float(sm_sh_align[0])
    bfc_hss_used = bfc_hss_angle      if bfc_hss_angle      is not None else float(sm_hss[0])
    action_type, is_mixed = classify_action(bfc_sh_used, bfc_hss_used)

    hss_at_release  = float(sm_hss[release_idx])
    peak_tlf        = float(np.percentile(sm_tlf, 95))
    knee_at_release = float(sm_knee_flex[release_idx])
    knee_risk       = knee_at_release < FRONT_KNEE_RISK
    ext_rate        = icc_extension / max(POST_WINDOW, 1)
    ext_rate_risk   = ext_rate > EXT_RATE_RISK
    min_interior    = float(np.min(sm_angles))
    extreme_elbow   = min_interior < MIN_INTERIOR_ANGLE_RISK

    risk_flags = {
        "Mixed bowling action (spinal twisting)":
            is_mixed,
        f"High hip-shoulder separation at release (above {HSS_MIXED_THRESHOLD} deg)":
            hss_at_release >= HSS_MIXED_THRESHOLD,
        f"High shoulder counter-rotation (above {SCR_MIXED_THRESHOLD} deg)":
            scr_max >= SCR_MIXED_THRESHOLD,
        f"Excessive trunk lean (above {TRUNK_LATERAL_RISK} deg)":
            peak_tlf > TRUNK_LATERAL_RISK,
        f"Low front knee flexion at FFC (below {FRONT_KNEE_RISK} deg)":
            knee_risk,
        f"Extreme elbow loading (below {MIN_INTERIOR_ANGLE_RISK} deg interior)":
            extreme_elbow,
        f"Rapid elbow snap (above {EXT_RATE_RISK} deg/frame)":
            ext_rate_risk,
    }

    score      = sum(risk_flags.values())
    risk_level = "LOW" if score == 0 else ("MODERATE" if score <= 2 else "HIGH")

    log("Generating charts...")
    progress(80)

    title_col  = '#e8e8ff'
    grid_col   = '#2a2a4a'
    release_lc = '#ffa500'
    icc_col    = '#00ff88' if icc_legal else '#ff4444'
    act_col    = '#ff4444' if is_mixed  else '#00ff88'

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#0d0d1a')
    fig.suptitle(
        f"BOWLING BIOMECHANICAL REPORT  —  {bowler_name.upper()}\n"
        f"ICC: {'LEGAL' if icc_legal else 'ILLEGAL'} ({icc_extension:.1f} deg)     "
        f"Action: {action_type}     Injury Risk: {risk_level} ({score}/{len(risk_flags)})",
        fontsize=12, fontweight='bold', color=title_col, y=0.98
    )

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    win_start = frames[max(0, release_idx - PRE_WINDOW)]
    win_end   = frames[min(len(frames)-1, release_idx + POST_WINDOW)]

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#111128')
        ax.tick_params(colors=title_col)
        ax.xaxis.label.set_color(title_col)
        ax.yaxis.label.set_color(title_col)
        ax.title.set_color(title_col)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_col)
        ax.grid(True, color=grid_col, alpha=0.6)
        ax.axvline(x=frames[release_idx], color=release_lc, linewidth=1.8,
                   linestyle='--', label='Release Frame', zorder=5)

    ax1.plot(frames, raw_angles, color='#444466', linestyle='--', linewidth=1,   label='Raw Signal')
    ax1.plot(frames, sm_angles,  color='#4488ff', linewidth=2.5,                 label='Smoothed Interior Angle')
    ax1.axhline(y=180, color='#ffffff', linestyle=':', alpha=0.3, linewidth=1)
    ax1.axvspan(win_start, win_end, alpha=0.12, color=release_lc, label='ICC Window')
    ax1.set_ylabel('Interior Angle (deg)', color=title_col)
    ax1.set_title(f'Elbow — ICC Extension: {icc_extension:.1f} deg  [{"LEGAL" if icc_legal else "ILLEGAL"}]',
                  color=icc_col, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a30', labelcolor=title_col, loc='upper right')
    ax1.set_ylim(60, 195)

    ax2.plot(frames, sm_hss,      color='#ff8844', linewidth=2.5, label='Hip-Shoulder Separation')
    ax2.plot(frames, sm_sh_align, color='#88ff44', linewidth=2.0, linestyle='-.', label='Shoulder Alignment')
    ax2.axhline(y=HSS_MIXED_THRESHOLD, color='#ff4444', linestyle=':', linewidth=1.5,
                label=f'Mixed Threshold: {HSS_MIXED_THRESHOLD} deg')
    ax2.set_ylabel('Angle (deg)', color=title_col)
    ax2.set_title(
        f'Action: {action_type}   |   HSS: {hss_at_release:.1f} deg   |   Max SCR: {scr_max:.1f} deg',
        color=act_col, fontweight='bold')
    ax2.legend(fontsize=8, facecolor='#1a1a30', labelcolor=title_col, loc='upper right')
    ax2.set_ylim(-5, 95)

    ax3.plot(frames, sm_tlf,       color='#ff44aa', linewidth=2.5, label='Trunk Lateral Flexion')
    ax3.plot(frames, sm_knee_flex, color='#44ddff', linewidth=2.0, linestyle='-.', label='Front Knee Flexion')
    ax3.axhline(y=TRUNK_LATERAL_RISK, color='#ff4444', linestyle=':', linewidth=1.5,
                label=f'Trunk Risk: {TRUNK_LATERAL_RISK} deg')
    ax3.axhline(y=FRONT_KNEE_RISK,    color='#ffaa00', linestyle=':', linewidth=1.5,
                label=f'Knee Risk: {FRONT_KNEE_RISK} deg')
    ax3.set_ylabel('Angle (deg)', color=title_col)
    ax3.set_xlabel('Frame Number', color=title_col)
    ax3.set_title(
        f'Peak Trunk Lean: {peak_tlf:.1f} deg   |   Front Knee: {knee_at_release:.1f} deg   |   '
        f'Elbow Snap: {ext_rate:.1f} deg/frame',
        color=title_col, fontweight='bold')
    ax3.legend(fontsize=8, facecolor='#1a1a30', labelcolor=title_col, loc='upper right')
    ax3.set_ylim(-5, 90)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("results", exist_ok=True)
    timestamp  = int(time.time())
    graph_path = os.path.join("results", f"report_{timestamp}.png")
    csv_path   = os.path.join("results", f"data_{timestamp}.csv")
    pdf_path   = os.path.join("results", f"report_{timestamp}.pdf")

    plt.savefig(graph_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()

    log("Saving CSV data...")
    progress(88)

    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['Frame', 'Raw_Elbow_Interior', 'Smoothed_Elbow_Interior',
                    'Flexion', 'HSS', 'Shoulder_Alignment', 'Trunk_Lat_Flex', 'Front_Knee_Flexion'])
        for i in range(len(frames)):
            w.writerow([
                frames[i], round(raw_angles[i], 2), round(sm_angles[i], 2),
                round(flexion[i], 2), round(sm_hss[i], 2), round(sm_sh_align[i], 2),
                round(sm_tlf[i], 2), round(sm_knee_flex[i], 2)
            ])

    log("Building PDF report...")
    progress(93)

    build_pdf_report(pdf_path, bowler_name, graph_path, {
        'icc_extension':  icc_extension,
        'icc_legal':      icc_legal,
        'action_type':    action_type,
        'is_mixed':       is_mixed,
        'bfc_sh_used':    bfc_sh_used,
        'bfc_hss_used':   bfc_hss_used,
        'hss_at_release': hss_at_release,
        'scr_max':        scr_max,
        'peak_tlf':       peak_tlf,
        'knee_at_release':knee_at_release,
        'min_interior':   min_interior,
        'ext_rate':       ext_rate,
        'risk_level':     risk_level,
        'score':          score,
        'total_flags':    len(risk_flags),
        'risk_flags':     risk_flags,
        'timestamp':      timestamp,
    })

    log("Analysis complete.")
    progress(100)

    return {
        'bowler_name':      bowler_name,
        'icc_extension':    round(icc_extension, 2),
        'icc_legal':        icc_legal,
        'action_type':      action_type,
        'is_mixed':         is_mixed,
        'hss_at_release':   round(hss_at_release, 1),
        'scr_max':          round(scr_max, 1),
        'peak_tlf':         round(peak_tlf, 1),
        'knee_at_release':  round(knee_at_release, 1),
        'min_interior':     round(min_interior, 1),
        'ext_rate':         round(ext_rate, 1),
        'risk_level':       risk_level,
        'score':            score,
        'total_flags':      len(risk_flags),
        'risk_flags':       {k: bool(v) for k, v in risk_flags.items()},
        'preview_frames':   preview_frames,
        'graph_path':       graph_path,
        'csv_path':         csv_path,
        'pdf_path':         pdf_path,
        'timestamp':        timestamp,
    }
