import streamlit as st
import tempfile
import cv2
import torch
import numpy as np
import time
import torch.nn as nn

from facenet_pytorch import MTCNN
from torchvision import models, transforms

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡",
    layout="wide"
)

# =========================================================
# MODERN UI CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
MAIN APP
========================================================= */

.stApp{
    background:#020617;
    color:white;
}

/* =========================================================
REMOVE STREAMLIT
========================================================= */

#MainMenu,
footer,
header{
    visibility:hidden;
}

/* =========================================================
CONTAINER
========================================================= */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1400px;
}

/* =========================================================
NAVBAR
========================================================= */

.navbar{
    display:flex;
    justify-content:space-between;
    align-items:center;

    padding:20px 30px;

    background:rgba(15,23,42,0.9);

    border:1px solid rgba(255,255,255,0.05);

    border-radius:20px;

    margin-bottom:30px;
}

/* =========================================================
LOGO
========================================================= */

.logo{
    font-size:30px;
    font-weight:800;
    color:white;
}

/* =========================================================
NAV ITEMS
========================================================= */

.nav-items{
    display:flex;
    gap:30px;

    color:#94a3b8;
    font-size:16px;
}

/* =========================================================
HERO
========================================================= */

.hero{
    text-align:center;
    padding:30px 20px;
}

/* =========================================================
TITLE
========================================================= */

.hero-title{
    font-size:58px;
    font-weight:800;
    color:white;
    line-height:1;
}

/* =========================================================
SUBTITLE
========================================================= */

.hero-sub{
    color:#94a3b8;
    font-size:22px;
    margin-top:20px;
}

/* =========================================================
MAIN CARD
========================================================= */

.main-card{
    background:rgba(15,23,42,0.85);

    border:1px solid rgba(255,255,255,0.05);

    border-radius:24px;

    padding:35px;

    margin-top:20px;
}

/* =========================================================
BUTTON
========================================================= */

div.stButton > button:first-child{

    width:100%;

    height:60px;

    border:none;

    border-radius:16px;

    background:linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    color:white;

    font-size:20px;

    font-weight:700;
}

/* =========================================================
RESULT BOX
========================================================= */

.real-box{

    background:linear-gradient(
        135deg,
        #052e16,
        #14532d
    );

    padding:30px;

    border-radius:20px;

    text-align:center;

    margin-top:30px;

    color:#4ade80;

    font-size:40px;

    font-weight:800;
}

.fake-box{

    background:linear-gradient(
        135deg,
        #450a0a,
        #7f1d1d
    );

    padding:30px;

    border-radius:20px;

    text-align:center;

    margin-top:30px;

    color:#f87171;

    font-size:40px;

    font-weight:800;
}

/* =========================================================
INFO TEXT
========================================================= */

.info-text{
    color:#cbd5e1;
    font-size:18px;
    margin-top:15px;
}

/* =========================================================
VIDEO
========================================================= */

video{
    border-radius:20px;
}

/* =========================================================
EXPANDER
========================================================= */

.streamlit-expanderHeader{
    font-size:18px;
    font-weight:600;
}

/* =========================================================
IMAGE
========================================================= */

img{
    border-radius:14px;
}

/* =========================================================
PROGRESS BAR
========================================================= */

.stProgress > div > div > div > div{
    background:#3b82f6;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# NAVBAR
# =========================================================

st.markdown("""
<div class="navbar">

<div class="logo">
🛡 DeepGuard AI
</div>

<div class="nav-items">
<div>Detection</div>
<div>Analysis</div>
<div>About</div>
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
Deepfake Detection
</div>

<div class="hero-sub">
Professional AI-powered video authenticity verification system
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cpu")

# =========================================================
# LSTM MODEL
# =========================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        output = self.fc(hidden[-1])

        return output

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = LSTMModel()

    model.load_state_dict(
        torch.load(
            "lstm_deepfake_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model

model = load_model()

# =========================================================
# FACE DETECTOR
# =========================================================

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# =========================================================
# CNN MODEL
# =========================================================

cnn_model = models.resnet18(pretrained=True)

cnn_model = torch.nn.Sequential(
    *list(cnn_model.children())[:-1]
)

cnn_model.eval()

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🏠 Detection",
    "📊 Analysis",
    "ℹ About"
])

# =========================================================
# VARIABLES
# =========================================================

real_percent = 0
fake_percent = 0
processing_time = 0
real_sequences = 0
fake_sequences = 0

# =========================================================
# TAB 1
# =========================================================

with tab1:

   
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "",
        type=["mp4", "avi", "mov"],
    label_visibility="collapsed"
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        detect = st.button("🚀 Detect Deepfake")

        if detect:

            start_time = time.time()

            progress = st.progress(0)

            status = st.empty()

            # =====================================================
            # SAVE VIDEO
            # =====================================================

            status.info("📥 Uploading Video...")
            progress.progress(10)

            temp_video = tempfile.NamedTemporaryFile(
                delete=False
            )

            temp_video.write(uploaded_file.read())

            video_path = temp_video.name

            # =====================================================
            # PROCESSING
            # =====================================================

            status.info("🧠 AI Analyzing Video...")
            progress.progress(30)

            cap = cv2.VideoCapture(video_path)

            frame_count = 0

            sequence = []

            st.session_state.frames_shown = []

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % 10 == 0:

                    display_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    st.session_state.frames_shown.append(
                        display_frame
                    )

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    boxes, _ = mtcnn.detect(rgb)

                    if boxes is not None:

                        for box in boxes:

                            x1, y1, x2, y2 = map(int, box)

                            x1 = max(0, x1)
                            y1 = max(0, y1)

                            face = frame[y1:y2, x1:x2]

                            if face.size == 0:
                                continue

                            h, w, _ = face.shape

                            if w < 80 or h < 80:
                                continue

                            face = cv2.resize(
                                face,
                                (224,224)
                            )

                            face = cv2.cvtColor(
                                face,
                                cv2.COLOR_BGR2RGB
                            )

                            tensor = transform(face)

                            tensor = tensor.unsqueeze(0)

                            with torch.no_grad():

                                features = cnn_model(tensor)

                            features = features.squeeze()

                            sequence.append(
                                features.numpy()
                            )

                            if len(sequence) == 10:

                                sequence_array = np.array(sequence)

                                sequence_tensor = torch.tensor(
                                    sequence_array,
                                    dtype=torch.float32
                                )

                                sequence_tensor = sequence_tensor.unsqueeze(0)

                                with torch.no_grad():

                                    output = model(sequence_tensor)

                                    prediction = torch.argmax(
                                        output,
                                        dim=1
                                    ).item()

                                if prediction == 0:
                                    real_sequences += 1
                                else:
                                    fake_sequences += 1

                                sequence = []

                frame_count += 1

            cap.release()

            progress.progress(100)

            status.success("✅ Detection Completed")

            total = real_sequences + fake_sequences

            if total > 0:

                real_percent = (
                    real_sequences / total
                ) * 100

                fake_percent = (
                    fake_sequences / total
                ) * 100

            processing_time = time.time() - start_time

            fake_ratio = fake_sequences / total if total > 0 else 0

            # =====================================================
            # FINAL RESULT
            # =====================================================

            if fake_ratio > 0.75:

                st.markdown(f"""
                <div class="fake-box">

                🚨 FAKE VIDEO DETECTED

                <div class="info-text">

                Confidence: {fake_percent:.2f}%<br><br>

                Processing Time: {processing_time:.2f} sec

                </div>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="real-box">

                ✅REAL VIDEO DETECTED

                <div class="info-text">

                Confidence: {real_percent:.2f}%<br><br>

                Processing Time: {processing_time:.2f} sec

                </div>

                </div>
                """, unsafe_allow_html=True)

    

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("📊 AI Analysis")

    # =====================================================
    # FRAMES
    # =====================================================

    with st.expander("🖼 View Extracted Frames"):

        if "frames_shown" in st.session_state:

            frames = st.session_state.frames_shown

            cols = st.columns(3)

            for idx, frame in enumerate(frames[:12]):

                with cols[idx % 3]:

                    st.image(
                        frame,
                        use_container_width=True
                    )

        else:

            st.info("No frames available yet.")

    # =====================================================
    # CONFIDENCE
    # =====================================================

    with st.expander("📈 Confidence Scores"):

        st.write(f"✅ Real Confidence: {real_percent:.2f}%")

        st.progress(int(real_percent))

        st.write(f"🚨 Fake Confidence: {fake_percent:.2f}%")

        st.progress(int(fake_percent))

    # =====================================================
    # PROCESSING INFO
    # =====================================================

    with st.expander("⚡ Processing Information"):

        st.write(f"Processing Time: {processing_time:.2f} sec")

        st.write(f"Real Sequences: {real_sequences}")

        st.write(f"Fake Sequences: {fake_sequences}")

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.markdown("""

    ## 🛡 About DeepGuard AI

    DeepGuard AI is a professional AI-powered
    deepfake detection system built using:

    - CNN (ResNet18)
    - LSTM Sequential Analysis
    - Face Detection using MTCNN
    - PyTorch Deep Learning
    - Streamlit Frontend

    ### Features

    ✅ AI-powered fake video detection  
    ✅ Face sequence analysis  
    ✅ Real/Fake confidence scoring  
    ✅ Extracted frame analysis  
    ✅ Modern SaaS-style dashboard  

    """)