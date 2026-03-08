"""
BowlScan — Streamlit GUI
Run with:  streamlit run streamlit_app.py
Requires:  bowling_analysis_engine.py in the same folder
"""

import os
import base64
import tempfile
import time

import streamlit as st
import numpy as np

from bowling_analysis_engine import run_analysis

# ---- Page config ----------------------------------------------------------------

st.set_page_config(
    page_title="BowlScan — Biomechanical Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Custom CSS -----------------------------------------------------------------

st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080810;
    color: #e8e8ff;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

/* ── Scanline overlay ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,200,255,0.012) 2px,
        rgba(0,200,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid #1e1e38;
    margin-bottom: 2rem;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #00c8ff;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.nav-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #00ff88;
    animation: pulse 2s infinite;
    display: inline-block;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(0.6); }
}
.nav-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #5a5a7a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border: 1px solid #1e1e38;
    padding: 4px 12px;
    border-radius: 2px;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00c8ff;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1e38;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e8e8ff;
    margin-bottom: 1.5rem;
}

/* ── Cards ── */
.card {
    background: #0e0e1c;
    border: 1px solid #1e1e38;
    border-radius: 6px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c8ff, #00ff88);
}

/* ── Verdict cards ── */
.verdict-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.verdict-card {
    flex: 1;
    background: #0e0e1c;
    border: 1px solid #1e1e38;
    border-radius: 6px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.verdict-card.legal    { border-top: 3px solid #00ff88; }
.verdict-card.illegal  { border-top: 3px solid #ff4444; }
.verdict-card.action   { border-top: 3px solid #00c8ff; }
.verdict-card.risk-low      { border-top: 3px solid #00ff88; }
.verdict-card.risk-moderate { border-top: 3px solid #ffaa00; }
.verdict-card.risk-high     { border-top: 3px solid #ff4444; }

.v-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 0.4rem;
}
.v-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.v-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #5a5a7a;
    margin-top: 0.3rem;
}
.color-legal   { color: #00ff88; }
.color-illegal { color: #ff4444; }
.color-accent  { color: #00c8ff; }
.color-warn    { color: #ffaa00; }

/* ── Metric tiles ── */
.metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.metric-tile {
    flex: 1;
    min-width: 140px;
    background: #0e0e1c;
    border: 1px solid #1e1e38;
    border-radius: 5px;
    padding: 0.9rem 0.8rem;
}
.metric-tile.ok      { border-left: 3px solid #00ff88; }
.metric-tile.flagged { border-left: 3px solid #ff4444; }
.metric-tile.warn    { border-left: 3px solid #ffaa00; }
.m-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 0.35rem;
    line-height: 1.4;
}
.m-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.0rem;
    font-weight: 700;
    color: #e8e8ff;
}
.m-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    margin-top: 0.25rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Risk checklist ── */
.risk-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e1e38;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #e8e8ff;
}
.risk-item:last-child { border-bottom: none; }
.risk-badge {
    flex-shrink: 0;
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.risk-badge.ok      { background: rgba(0,255,136,0.1); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
.risk-badge.flagged { background: rgba(255,68,68,0.1);  color: #ff4444; border: 1px solid rgba(255,68,68,0.3);  }

/* ── Hint cards ── */
.hint-row { display: flex; gap: 0.8rem; margin-top: 1rem; flex-wrap: wrap; }
.hint {
    flex: 1;
    min-width: 180px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #5a5a7a;
    background: #0e0e1c;
    border: 1px solid #1e1e38;
    border-left: 2px solid #00c8ff;
    padding: 0.5rem 0.8rem;
    border-radius: 2px;
    line-height: 1.5;
}

/* ── Progress ── */
.progress-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00c8ff;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] {
    background: #0e0e1c;
    border: 2px dashed #2a2a4a;
    border-radius: 6px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #00c8ff;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
    background: #0e0e1c !important;
    border-color: #2a2a4a !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(0,255,136,0.1));
    border: 1px solid #00c8ff;
    color: #00c8ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 0.7rem 1.5rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,200,255,0.28), rgba(0,255,136,0.18));
    box-shadow: 0 0 20px rgba(0,200,255,0.2);
    transform: translateY(-1px);
}
.stDownloadButton > button {
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    width: 100%;
    padding: 0.65rem;
}
label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #5a5a7a !important;
}
.stAlert {
    background: rgba(255,68,68,0.06) !important;
    border: 1px solid rgba(255,68,68,0.3) !important;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.stSuccess {
    background: rgba(0,255,136,0.06) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
/* Number input */
div[data-testid="stNumberInput"] input {
    background: #0e0e1c !important;
    border-color: #2a2a4a !important;
    color: #e8e8ff !important;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ---- Helper: render HTML blocks -------------------------------------------------

def nav_bar():
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">
        <span class="nav-dot"></span>
        BOWLSCAN
      </div>
      <span class="nav-tag">Biomechanical Analysis System</span>
    </div>
    """, unsafe_allow_html=True)


def section_header(label, title):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def verdict_cards(r):
    icc_class   = "legal"   if r['icc_legal'] else "illegal"
    icc_color   = "color-legal" if r['icc_legal'] else "color-illegal"
    icc_verdict = "LEGAL"   if r['icc_legal'] else "ILLEGAL"

    risk_class = f"risk-{r['risk_level'].lower()}"
    risk_color = "color-legal" if r['risk_level'] == "LOW" else (
                 "color-warn"  if r['risk_level'] == "MODERATE" else "color-illegal")

    st.markdown(f"""
    <div class="verdict-row">
      <div class="verdict-card {icc_class}">
        <div class="v-label">ICC Elbow Verdict</div>
        <div class="v-value {icc_color}">{icc_verdict}</div>
        <div class="v-sub">{r['icc_extension']} deg extension at release</div>
      </div>
      <div class="verdict-card action">
        <div class="v-label">Bowling Action</div>
        <div class="v-value color-accent" style="font-size:1.15rem">{r['action_type']}</div>
        <div class="v-sub">Classified at back foot contact</div>
      </div>
      <div class="verdict-card {risk_class}">
        <div class="v-label">Injury Risk</div>
        <div class="v-value {risk_color}">{r['risk_level']}</div>
        <div class="v-sub">{r['score']} of {r['total_flags']} risk flags triggered</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def metric_tiles(r):
    metrics = [
        ("ICC Extension",        f"{r['icc_extension']} deg",     r['icc_legal'],                      "Limit: 15 deg"),
        ("Hip-Shoulder Sep.",    f"{r['hss_at_release']} deg",    r['hss_at_release'] < 20,            "At release"),
        ("Shoulder Counter-Rot", f"{r['scr_max']} deg",           r['scr_max'] < 30,                   "Max post-BFC"),
        ("Peak Trunk Lean",      f"{r['peak_tlf']} deg",          r['peak_tlf'] <= 45,                 "Limit: 45 deg"),
        ("Front Knee at FFC",    f"{r['knee_at_release']} deg",   r['knee_at_release'] >= 13,          "Min: 13 deg"),
        ("Min Interior Angle",   f"{r['min_interior']} deg",      r['min_interior'] >= 120,            "UCL threshold: 120 deg"),
        ("Elbow Snap Rate",      f"{r['ext_rate']} deg/frame",    r['ext_rate'] <= 8,                  "Limit: 8 deg/frame"),
        ("Action Type",          r['action_type'].split(' ')[0],  not r['is_mixed'],                   "Mixed" if r['is_mixed'] else "No mixed action"),
    ]

    tiles_html = '<div class="metrics-row">'
    for label, value, ok, sub in metrics:
        tile_class  = "ok" if ok else "flagged"
        stat_color  = "#00ff88" if ok else "#ff4444"
        stat_text   = "Normal" if ok else "Flagged"
        tiles_html += f"""
        <div class="metric-tile {tile_class}">
          <div class="m-label">{label}</div>
          <div class="m-value">{value}</div>
          <div class="m-status" style="color:{stat_color}">{stat_text}
            <span style="color:#5a5a7a"> · {sub}</span>
          </div>
        </div>"""
    tiles_html += '</div>'
    st.markdown(tiles_html, unsafe_allow_html=True)


def risk_checklist(r):
    items_html = ''
    for label, triggered in r['risk_flags'].items():
        badge_class = "flagged" if triggered else "ok"
        badge_text  = "Flagged" if triggered else "OK"
        items_html += f"""
        <div class="risk-item">
          <span class="risk-badge {badge_class}">{badge_text}</span>
          <span>{label}</span>
        </div>"""

    st.markdown(f"""
    <div class="card" style="margin-bottom:1.5rem">
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                  text-transform:uppercase;color:#00c8ff;margin-bottom:1rem;
                  padding-bottom:0.6rem;border-bottom:1px solid #1e1e38;">
        Injury Risk Checklist
      </div>
      {items_html}
    </div>
    """, unsafe_allow_html=True)


# ---- Session state init ---------------------------------------------------------

if 'results' not in st.session_state:
    st.session_state.results = None
if 'log_lines' not in st.session_state:
    st.session_state.log_lines = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False


# ---- Main layout ----------------------------------------------------------------

nav_bar()

# ===== RESULTS VIEW =============================================================
if st.session_state.analysis_done and st.session_state.results:
    r = st.session_state.results

    col_head, col_btn = st.columns([4, 1])
    with col_head:
        section_header("Analysis Complete", f"{r['bowler_name']} — Biomechanical Report")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↩  New Analysis"):
            st.session_state.analysis_done = False
            st.session_state.results = None
            st.session_state.log_lines = []
            st.rerun()

    # Three verdict cards
    verdict_cards(r)

    # Graph
    st.markdown('<div class="section-label">Biomechanical Trace — Full Delivery Arc</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:0;margin-bottom:1.5rem">', unsafe_allow_html=True)
    st.image(r['graph_path'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Metric tiles
    st.markdown('<div class="section-label">Detailed Metrics</div>', unsafe_allow_html=True)
    metric_tiles(r)

    # Risk checklist
    st.markdown('<div class="section-label">Injury Risk Assessment</div>', unsafe_allow_html=True)
    risk_checklist(r)

    # Skeleton preview frames
    if r.get('preview_frames'):
        st.markdown('<div class="section-label">Skeleton Tracking Preview</div>', unsafe_allow_html=True)
        frames_to_show = r['preview_frames'][-12:]
        cols = st.columns(min(6, len(frames_to_show)))
        for i, frame_b64 in enumerate(frames_to_show):
            img_bytes = base64.b64decode(frame_b64)
            with cols[i % len(cols)]:
                st.image(img_bytes, use_container_width=True)

    # Downloads
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Export Results</div>', unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        if r.get('csv_path') and os.path.exists(r['csv_path']):
            with open(r['csv_path'], 'rb') as f:
                st.download_button(
                    label="⬇  Download CSV Data",
                    data=f,
                    file_name=f"bowlscan_{r['bowler_name'].replace(' ','_')}.csv",
                    mime='text/csv',
                    use_container_width=True,
                )

    with dl_col2:
        if r.get('pdf_path') and os.path.exists(r['pdf_path']):
            with open(r['pdf_path'], 'rb') as f:
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=f,
                    file_name=f"bowlscan_{r['bowler_name'].replace(' ','_')}.pdf",
                    mime='application/pdf',
                    use_container_width=True,
                )

    with dl_col3:
        if r.get('graph_path') and os.path.exists(r['graph_path']):
            with open(r['graph_path'], 'rb') as f:
                st.download_button(
                    label="⬇  Download Graph PNG",
                    data=f,
                    file_name=f"bowlscan_graph_{r['bowler_name'].replace(' ','_')}.png",
                    mime='image/png',
                    use_container_width=True,
                )


# ===== UPLOAD / INPUT VIEW ======================================================
else:
    section_header("New Session", "Bowler Analysis")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Row 1: name + arm
    col1, col2 = st.columns(2)
    with col1:
        bowler_name = st.text_input("Bowler Name", placeholder="e.g. Arjun Sharma", value="")
    with col2:
        bowler_arm = st.selectbox("Bowling Arm", ["RIGHT", "LEFT"])

    # Row 2: frame trim
    col3, col4 = st.columns(2)
    with col3:
        start_frame = st.number_input("Start Frame (optional trim)", min_value=0, value=0, step=1)
    with col4:
        end_frame = st.number_input("End Frame (optional trim)", min_value=1, value=999999, step=1)

    # Video uploader
    uploaded = st.file_uploader(
        "Video File",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        help="Side-on camera angle required. Max 500 MB."
    )

    # Hint cards
    st.markdown("""
    <div class="hint-row">
      <div class="hint">Camera must be side-on (perpendicular to the pitch) for accurate elbow angle measurement.</div>
      <div class="hint">Film in daylight. The bowler should fill at least 40% of the frame height during delivery.</div>
      <div class="hint">Trim to a single delivery using Start / End Frame if the video contains multiple deliveries or a long run-up.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Preview section -------------------------------------------------------
    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        col_prev, col_run = st.columns([1, 1])

        with col_prev:
            if st.button("▶  Play Skeleton Preview"):
                st.session_state.show_preview = True

        with col_run:
            run_clicked = st.button("▶  Run Full Analysis")
        
        # Show preview
        if st.session_state.show_preview and not run_clicked:
            import cv2
            import mediapipe as mp_lib

            st.markdown('<div class="section-label" style="margin-top:1rem">Skeleton Preview</div>',
                        unsafe_allow_html=True)

            tfile_prev = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile_prev.write(uploaded.getvalue())
            tfile_prev.flush()

            cap_prev = cv2.VideoCapture(tfile_prev.name)
            mp_pose_p = mp_lib.solutions.pose
            pose_prev = mp_pose_p.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_draw   = mp_lib.solutions.drawing_utils
            mp_styles = mp_lib.solutions.drawing_styles

            frame_slot = st.empty()
            stop_btn   = st.button("■  Stop Preview")
            frame_num  = 0

            while cap_prev.isOpened() and not stop_btn:
                ret, frame = cap_prev.read()
                if not ret:
                    break
                frame_num += 1
                if frame_num % 2 != 0:  # skip every other frame for speed
                    continue

                frame_small = cv2.resize(frame, (800, 450))
                rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                res = pose_prev.process(rgb)
                out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if res.pose_landmarks:
                    mp_draw.draw_landmarks(
                        out, res.pose_landmarks, mp_pose_p.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )

                frame_slot.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap_prev.release()
            os.unlink(tfile_prev.name)
            st.session_state.show_preview = False

        # ---- Run full analysis -------------------------------------------------
        if run_clicked:
            if not uploaded:
                st.error("Please select a video file first.")
            else:
                name = bowler_name.strip() or "Unknown Bowler"

                # Save upload to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.getvalue())
                tfile.flush()
                video_path = tfile.name

                st.markdown('<div class="section-label" style="margin-top:1rem">Processing</div>',
                            unsafe_allow_html=True)

                progress_bar = st.progress(0)
                status_text  = st.empty()
                log_box      = st.empty()

                log_lines = []

                def on_progress(pct):
                    progress_bar.progress(min(pct, 100))

                def on_log(msg):
                    log_lines.append(msg)
                    log_box.markdown(
                        "<div style='background:#080810;border:1px solid #1e1e38;border-radius:4px;"
                        "padding:0.8rem 1rem;font-family:Space Mono,monospace;font-size:0.68rem;"
                        "color:#5a5a7a;max-height:140px;overflow-y:auto;line-height:1.7'>"
                        + "".join(
                            f"<div style='color:{'#ff4444' if l.startswith('ERROR') else '#e8e8ff' if i==len(log_lines)-1 else '#5a5a7a'}'>{l}</div>"
                            for i, l in enumerate(log_lines)
                        )
                        + "</div>",
                        unsafe_allow_html=True
                    )

                try:
                    status_text.markdown(
                        '<span style="font-family:Space Mono,monospace;font-size:0.72rem;'
                        'color:#00c8ff">Analysing…</span>',
                        unsafe_allow_html=True
                    )
                    results = run_analysis(
                        video_path    = video_path,
                        bowler_arm    = bowler_arm,
                        bowler_name   = name,
                        start_frame   = int(start_frame),
                        end_frame     = int(end_frame),
                        progress_callback = on_progress,
                        log_callback  = on_log,
                    )

                    progress_bar.progress(100)
                    status_text.markdown(
                        '<span style="font-family:Space Mono,monospace;font-size:0.72rem;'
                        'color:#00ff88">Analysis complete.</span>',
                        unsafe_allow_html=True
                    )

                    st.session_state.results      = results
                    st.session_state.log_lines    = log_lines
                    st.session_state.analysis_done = True

                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.empty()
                    st.error(f"Analysis failed: {e}")

                finally:
                    try:
                        os.unlink(video_path)
                    except Exception:
                        pass
