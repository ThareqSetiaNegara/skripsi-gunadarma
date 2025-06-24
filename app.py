import streamlit as st
import sqlite3
from datetime import datetime
import io
import cv2
import av
import PIL.Image as Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import settings  # Asumsi file settings.py ada dan berisi DEFAULT_IMAGE, DEFAULT_DETECT_IMAGE, DETECTION_MODEL
from ultralytics import YOLO
import google.generativeai as genai
import os
from google import generativeai as genai
from fpdf import FPDF
import tempfile

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def clean_markdown(text):
    """Membersihkan format markdown dari teks untuk output PDF."""
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('#', '')
    text = text.replace('`', '')
    return text


def create_detection_pdf(image, label, confidence, explanation):
    """
    Membuat file PDF yang berisi hasil deteksi, gambar, dan penjelasan.
    """
    try:
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font('Arial', 'B', 18)
        pdf.cell(190, 10, 'Hasil Deteksi Penyakit Daun Padi', 0, 1, 'C')
        pdf.ln(10)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(190, 10, f'Waktu Deteksi: {current_time}', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 14)
        # Mengubah format confidence menjadi persentase
        pdf.cell(190, 10, f'Penyakit Terdeteksi: {label}', 0, 1)
        pdf.cell(190, 10, f'Tingkat Kepercayaan: {confidence:.0%}', 0, 1) # Mengubah format ke persen
        pdf.ln(5)

        # Simpan gambar sementara untuk dimasukkan ke PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_filename = temp_file.name
            image.save(temp_file, format='PNG')  # Menggunakan temp_file langsung

        pdf.cell(190, 10, 'Gambar Daun Padi:', 0, 1)
        pdf.image(temp_filename, x=10, y=None, w=180)
        os.unlink(temp_filename)  # Hapus file sementara

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, 'Analisis dan Rekomendasi:', 0, 1)
        pdf.ln(5)

        pdf.set_font('Arial', '', 14)
        explanation_lines = explanation.split('\n')

        current_mode = 'normal'
        for line in explanation_lines:
            clean_line = clean_markdown(line)

            # Deteksi judul bagian dalam penjelasan
            if "Penjelasan:" in clean_line or "Dampak:" in clean_line or "Rekomendasi" in clean_line:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 14)
                current_mode = 'title'
            elif clean_line.strip() == "":
                pdf.ln(5)
                pdf.set_font('Arial', '', 14)
                current_mode = 'normal'
            else:
                if current_mode == 'title':
                    pdf.set_font('Arial', '', 14)
                    current_mode = 'normal'

            pdf.multi_cell(0, 6, clean_line)

        # Simpan PDF sementara dan baca isinya
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
            temp_pdf_filename = temp_pdf_file.name
            pdf.output(temp_pdf_filename)

        with open(temp_pdf_filename, 'rb') as f:
            pdf_data = f.read()

        os.unlink(temp_pdf_filename)  # Hapus file PDF sementara
        return pdf_data

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat PDF: {str(e)}")
        return None


# Model untuk deteksi objek dengan webcam
class VideoTransformer(VideoProcessorBase):
    """
    Kelas pemroses video untuk deteksi objek real-time menggunakan webcam.
    """

    def __init__(self):
        self.model = YOLO(settings.DETECTION_MODEL)
        self.confidence = 0.3
        self.detected_objects = []
        self.resize_dim = None # Default: no resize

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Menerima frame video, melakukan deteksi, dan mengembalikan frame dengan kotak pembatas.
        """
        img = frame.to_ndarray(format="bgr24")
        self.detected_objects = []

        # Resize frame jika resize_dim diatur
        if self.resize_dim:
            img_resized = cv2.resize(img, self.resize_dim)
        else:
            img_resized = img

        results = self.model(img_resized, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                c = box.cls
                conf = box.conf.item()
                if conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, b)
                    # Mengubah format confidence menjadi persentase
                    label = f"{self.model.names[int(c)]} {conf:.0%}" # Mengubah format ke persen

                    # Jika frame di-resize sebelum deteksi, koordinat bounding box perlu diskalakan kembali
                    if self.resize_dim:
                        original_h, original_w, _ = img.shape
                        resized_w, resized_h = self.resize_dim
                        scale_x = original_w / resized_w
                        scale_y = original_h / resized_h
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)


                    # Define font properties for the label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.8  # Ukuran font 2x lebih besar dari 0.9
                    font_thickness = 2
                    text_color = (255, 255, 255)  # Putih
                    bg_color = (0, 0, 0)  # Hitam

                    # Get text size to calculate background rectangle dimensions
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, font_thickness)

                    # Calculate the top-left corner of the text's baseline (original position)
                    text_baseline_y = y1 - 10
                    # Calculate the actual top of the text based on its height
                    text_top_y = text_baseline_y - text_height

                    # Add some padding around the text for the background rectangle
                    padding_x = 10
                    padding_y = 5

                    # Coordinates for the black background rectangle
                    # Start from x1, and slightly above the text top
                    bg_rect_x1 = x1
                    bg_rect_y1 = text_top_y - padding_y
                    bg_rect_x2 = x1 + text_width + padding_x * 2
                    bg_rect_y2 = text_baseline_y + baseline + padding_y  # Extend slightly below baseline

                    # Ensure coordinates are within image bounds to prevent drawing outside
                    bg_rect_x1 = max(0, bg_rect_x1)
                    bg_rect_y1 = max(0, bg_rect_y1)
                    bg_rect_x2 = min(img.shape[1], bg_rect_x2)
                    bg_rect_y2 = min(img.shape[0], bg_rect_y2)

                    # Draw the filled black background rectangle
                    cv2.rectangle(
                        img, (bg_rect_x1, bg_rect_y1), (bg_rect_x2, bg_rect_y2), bg_color, -1)

                    # Draw the bounding box (green color, as it was)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw the text (white color) on top of the black background
                    # The text's x position should be adjusted for padding within the background rectangle
                    # The text's baseline y position remains the same as calculated before
                    cv2.putText(img, label, (x1 + padding_x, text_baseline_y),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    self.detected_objects.append({
                        'label': self.model.names[int(c)],
                        'confidence': conf,
                        'box': b
                    })

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Fungsi untuk halaman deteksi (sebelumnya main_app)
def detection_page():
    """
    Fungsi untuk halaman deteksi penyakit daun padi.
    """
    try:
        # Mengambil API key Gemini dari Streamlit secrets
        gemini_api_key = st.secrets["gemini"]["api_key"]
        genai.configure(api_key=gemini_api_key)
        GEMINI_CONFIGURATED = True
    except Exception as e:
        GEMINI_CONFIGURATED = False
        print(f"Error konfigurasi Gemini API: {str(e)}")

    def get_disease_explanation(disease_label):
        """
        Mendapatkan penjelasan detail tentang penyakit dari model Gemini.
        """
        if not GEMINI_CONFIGURATED:
            return "API Gemini belum terkonfigurasi dengan benar. Silakan periksa konfigurasi API key Anda."

        try:
            prompt = f"""
            Berikan penjelasan detail tentang penyakit daun padi "{disease_label}" dengan format berikut:
            
            PENJELASAN:
            [Jelaskan gejala dan penyebab penyakit pada daun padi tersebut secara detail]
            
            DAMPAK:
            [Jelaskan dampak penyakit ini terhadap tanaman daun padi]
            
            REKOMENDASI PENANGANAN:
            [Berikan 2-4 rekomendasi penanganan yang bisa dilakukan petani]
            """

            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Terjadi kesalahan saat mendapatkan penjelasan dari Gemini: {str(e)}"

    # Inisialisasi state sesi jika belum ada
    if 'detection_boxes' not in st.session_state:
        st.session_state.detection_boxes = None
    if 'detection_model' not in st.session_state:
        st.session_state.detection_model = None
    if 'detection_confidence' not in st.session_state:
        st.session_state.detection_confidence = None

    def save_detection(image):
        """
        Menyimpan gambar hasil deteksi ke database SQLite.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        conn.execute(
            "INSERT INTO detections (timestamp, image) VALUES (?, ?)", (timestamp, img_byte_arr))
        conn.commit()

    def load_detection_history():
        """
        Memuat riwayat deteksi dari database SQLite.
        """
        c = conn.cursor()
        c.execute(
            "SELECT id, timestamp, image FROM detections ORDER BY timestamp DESC")
        return c.fetchall()

    def delete_all_detections():
        """
        Menghapus semua riwayat deteksi dari database SQLite.
        """
        c = conn.cursor()
        c.execute("DELETE FROM detections")
        conn.commit()

    # Inisialisasi database SQLite
    conn = sqlite3.connect(
        'detection_paddy_leaves.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  image BLOB)''')
    conn.commit()

    st.title("Deteksi Penyakit Tanaman Daun Padi")
    st.markdown("---")

    history_placeholder = st.empty()

    # Kontrol Sidebar
    st.sidebar.header("Pengaturan Deteksi")
    confidence = float(st.sidebar.slider(
        "Pilih Tingkat Kepercayaan Model (%)", 25, 100, 30)) / 100

    # Asumsi settings.DETECTION_MODEL mengarah ke path model YOLO
    model_path = os.path.join(os.getcwd(), 'weights', 'best.pt')
    model = YOLO(model_path)

    st.sidebar.header("Konfigurasi Gambar/Video")
    source_radio = st.sidebar.radio(
        "Pilih Sumber", ["Unggah Gambar", "Kamera"])  # Menggunakan string langsung untuk kemudahan

    source_img = None

    # Deteksi Gambar
    if source_radio == "Unggah Gambar":
        source_img = st.sidebar.file_uploader(
            "", type=("jpg", "jpeg", "png", 'bmp', 'webp'))  # Label diatur menjadi string kosong

        detect_button = st.sidebar.button('🔍 Deteksi Objek')

        # Layout berdampingan untuk gambar asli dan hasil deteksi
        col1, col2 = st.columns(2, gap="medium") # Mengubah gap antar kolom menjadi "medium"

        with col1:
            st.subheader("📷 Gambar Asli")
            try:
                if source_img is None:
                    # Asumsi settings.DEFAULT_IMAGE adalah path ke gambar default
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = Image.open(default_image_path)
                    st.image(default_image_path, caption="Gambar Default",
                            use_column_width=True)
                else:
                    uploaded_image = Image.open(source_img)
                    st.image(source_img, caption="Gambar yang Diunggah",
                            use_column_width=True)
            except Exception as ex:
                st.error(
                    "Terjadi kesalahan saat membuka gambar. Pastikan file adalah gambar yang valid.")
                st.error(ex)

        with col2:
            st.subheader("🎯 Hasil Deteksi")
            if source_img is None:
                # Asumsi settings.DEFAULT_DETECT_IMAGE adalah path ke gambar deteksi default
                default_detected_image_path = str(
                    settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path,
                        caption='Gambar Terdeteksi Default',
                        use_column_width=True)
            else:
                if detect_button:
                    with st.spinner("⏳ Melakukan deteksi objek..."):
                        res = model.predict(uploaded_image,
                                            conf=confidence
                                            )
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        detected_image = Image.fromarray(res_plotted)
                        st.image(res_plotted, caption='Gambar Terdeteksi',
                                use_column_width=True)
                        save_detection(detected_image)  # Simpan hasil deteksi

                        st.session_state.detection_boxes = boxes
                        st.session_state.detection_model = model
                        st.session_state.detection_confidence = confidence

        # Tampilkan analisis deteksi jika ada hasil dan tombol deteksi ditekan
        if source_img is not None and detect_button and 'detection_boxes' in st.session_state:
            st.markdown("---")
            st.header("📊 Hasil Analisis Deteksi")

            boxes = st.session_state.detection_boxes
            model = st.session_state.detection_model
            confidence = st.session_state.detection_confidence

            if len(boxes) == 0:
                st.info(
                    "Tidak ada penyakit daun padi yang terdeteksi pada gambar ini dengan tingkat kepercayaan yang dipilih.")
            else:
                for box in boxes:
                    label = model.names[int(box.cls)]
                    conf = box.conf.item()
                    with st.container():
                        st.subheader(
                            f"Deteksi: {label} (Kepercayaan: {conf:.0%})") # Mengubah format ke persen

                        if GEMINI_CONFIGURATED:
                            with st.spinner(f"🔄 Mendapatkan penjelasan untuk '{label}'..."):
                                explanation = get_disease_explanation(
                                    label)
                                st.markdown(explanation)

                                col1, col2 = st.columns([1, 6])
                                with col1:
                                    pdf_data = create_detection_pdf(
                                        detected_image, label, conf, explanation)
                                    if pdf_data:
                                        filename = f"deteksi_{label.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                        st.download_button(
                                            label="📥 Unduh Laporan PDF",
                                            data=pdf_data,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_{label}_{conf}"
                                        )
                        else:
                            st.warning(
                                "⚠️ API Gemini tidak terkonfigurasi. Penjelasan penyakit tidak dapat ditampilkan.")

                        st.markdown("---")

    # Deteksi Webcam
    elif source_radio == "Kamera":
        st.header("📹 Deteksi Penyakit via Webcam")
        st.warning(
            "⚠️ Tekan tombol START di bawah untuk memulai webcam. Pastikan Anda mengizinkan akses kamera.")

        # Tambahkan kontrol untuk resolusi di sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Optimasi Webcam")
        resize_option = st.sidebar.selectbox(
            "Resolusi Proses Webcam",
            ["Original", "640x480", "480x360", "320x240"],
            index=1, # Default ke 640x480
            help="Mengubah ukuran frame sebelum deteksi. Resolusi lebih rendah = performa lebih baik."
        )

        resize_dim_tuple = None
        if resize_option != "Original":
            width, height = map(int, resize_option.split('x'))
            resize_dim_tuple = (width, height)

        # Inisialisasi webcam streamer
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Atur tingkat kepercayaan dan resolusi untuk pemroses video jika sudah aktif
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence = confidence
            webrtc_ctx.video_processor.resize_dim = resize_dim_tuple


    # Riwayat Deteksi
    if st.sidebar.button('📚 Lihat Riwayat Deteksi'):
        with history_placeholder.container():
            st.header("📚 Riwayat Deteksi")
            if st.button("❌ Tutup Riwayat"):
                history_placeholder.empty()

            history = load_detection_history()

            if not history or len(history) == 0:
                st.info("ℹ️ Belum ada riwayat deteksi yang tersimpan.")
            else:
                st.success(f"✅ Ditemukan {len(history)} hasil deteksi.")
                for id, timestamp, image in history:
                    try:
                        image = Image.open(io.BytesIO(image))
                        with st.expander(f"🆔 ID: {id}, ⏰ Waktu: {timestamp}"):
                            st.image(image, caption="Gambar Terdeteksi",
                                    use_column_width=False, width=500)
                    except Exception as e:
                        st.error(f"❌ Error menampilkan gambar ID: {id}: {str(e)}")

    if st.sidebar.button('🗑️ Hapus Semua Riwayat'):
        delete_all_detections()
        st.sidebar.success("✅ Semua riwayat deteksi telah dihapus.")
        history_placeholder.empty()  # Kosongkan placeholder riwayat
        with history_placeholder.container():  # Tampilkan pesan kosong setelah dihapus
            st.header("📚 Riwayat Deteksi")
            st.info("ℹ️ Tidak ada riwayat deteksi tersedia.")

# Fungsi untuk halaman utama (homepage)
def homepage():
    """
    Fungsi untuk menampilkan halaman utama aplikasi dengan panduan penggunaan,
    termasuk gambar di setiap langkah.
    """
    st.title("Selamat Datang di Aplikasi Deteksi Penyakit Daun Padi")
    st.markdown("---")

    st.markdown("""
    <p style='font-size: 32px; text-align: center; color: #1a1a1a; margin-bottom: 1rem;'>
        Aplikasi ini dirancang untuk membantu petani maupun masyarakat umum dalam mendeteksi penyakit pada daun padi secara cepat dan akurat.
        Dengan teknologi deteksi objek terkini dan dukungan AI, Anda dapat mendeteksi masalah pada tanaman padi Anda
        dan mendapatkan rekomendasi penanganan yang tepat.
    </p>
    """, unsafe_allow_html=True)

    st.header("Panduan Penggunaan Aplikasi")
    st.markdown("""
    <p style='font-size: 28px; line-height: 1.5; color: #1a1a1a;'>
        Ikuti langkah-langkah di bawah ini untuk memulai deteksi:
    </p>
    """, unsafe_allow_html=True)

    st.subheader("1. Membuka Halaman Deteksi")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Langkah pertama adalah mengakses halaman deteksi pada aplikasi.
    Ini dilakukan dengan mengklik tombol yang sesuai untuk menuju halaman deteksi dari halaman utama aplikasi. Tombol terlihat pada menu samping
    ataupun di bawah pada halaman ini
    </p>
    """, unsafe_allow_html=True)
    image_path_1 = "assets/guide/1.png"
    if not os.path.exists(image_path_1):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_1}")
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1]) # Rasio kolom disesuaikan untuk pemusatan
    with col2:
        st.image(image_path_1, caption="Gambar: Halaman Utama dengan Tombol Deteksi", use_column_width=True, width=10) # Menambahkan width

    st.subheader("2. Pengaturan Deteksi (Opsional)")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Setelah berada di halaman deteksi, Anda akan melihat bagian "Pengaturan Deteksi".
    Di bagian ini, terdapat opsi untuk mengatur tingkat kepercayaan model deteksi, yang mana merupakan nilai minimum yang ingin anda capai.
    Namun, disarankan untuk <b>mengabaikan</b> pengaturan ini agar memperoleh hasil yang maksimal dalam melakukan deteksi.
    </p>
    """, unsafe_allow_html=True)
    image_path_2 = "assets/guide/2.png"
    if not os.path.exists(image_path_2):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_2}")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        st.image(image_path_2, caption="Gambar: Pengaturan Deteksi di Sidebar", use_column_width=True, width=10)

    st.subheader("3. Memilih Metode Deteksi")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Selanjutnya, Anda akan menemukan bagian "Konfigurasi Metode Deteksi".
    Di sini, Anda perlu memilih sumber gambar yang akan digunakan untuk deteksi.
    Terdapat dua pilihan sumber:
    <ul>
        <li><b>Unggah Gambar:</b> Pilih opsi ini untuk mengunggah gambar daun padi dari perangkat Anda.</li>
        <li><b>Kamera:</b> Pilih opsi ini untuk menggunakan kamera perangkat Anda secara langsung.</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)
    image_path_3 = "assets/guide/3.png"
    if not os.path.exists(image_path_3):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_3}")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        st.image(image_path_3, caption="Gambar: Pilihan Sumber Gambar/Video", use_column_width=True, width=10)

    st.subheader("4. Memilih Gambar (Jika Menggunakan Unggah Gambar)")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Jika Anda memilih "Unggah Gambar", area untuk mengunggah file akan ditampilkan.
    Anda dapat mengunggah gambar dengan salah satu cara berikut:
    <ul>
        <li>Seret dan letakkan file gambar ke area yang ditentukan.</li>
        <li>Klik tombol "Browse files" untuk memilih file gambar dari penyimpanan perangkat Anda.</li>
    </ul>
    Aplikasi mendukung format file JPG, JPEG, PNG, BMP, dan WEBP dengan batasan ukuran file maksimum (misalnya, 200MB).
    </p>
    """, unsafe_allow_html=True)
    # Memusatkan dan mengecilkan gambar untuk langkah 4 menggunakan st.columns
    image_path_4 = "assets/guide/4.png"
    if not os.path.exists(image_path_4):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_4}")
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col2:
        st.image(image_path_4, caption="Gambar: Area Unggah Gambar", use_column_width=True, width=10)

    st.subheader("5. Melakukan Deteksi")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Setelah gambar dipilih dan terunggah (jika menggunakan metode unggah gambar), atau setelah kamera diaktifkan, proses deteksi perlu dimulai.
    <ul>
        <li>Jika menggunakan metode unggah gambar, klik tombol "Deteksi Objek" untuk memulai analisis gambar.</li>
        <li>Jika menggunakan kamera, deteksi mungkin dilakukan secara otomatis setelah kamera menangkap gambar.</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)
    # Memusatkan dan mengecilkan gambar untuk langkah 5 menggunakan st.columns
    image_path_5 = "assets/guide/5.png"
    if not os.path.exists(image_path_5):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_5}")
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    with col2:
        st.image(image_path_5, caption="Gambar: Tombol Deteksi Objek atau Tampilan Kamera Aktif", use_column_width=True, width=10)

    st.subheader("6. Melihat Hasil Deteksi")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Setelah proses deteksi selesai, hasil deteksi akan ditampilkan.
    Hasil ini mencakup:
    <ul>
        <li>Gambar asli daun padi.</li>
        <li>Hasil analisis deteksi yang memberikan informasi mengenai jenis penyakit yang terdeteksi (jika ada) dan tingkat kepercayaan dari deteksi tersebut.</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)
    # Memusatkan dan mengecilkan gambar untuk langkah 6 menggunakan st.columns
    image_path_6 = "assets/guide/6.png"
    if not os.path.exists(image_path_6):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_6}")
    image_path_7 = "assets/guide/7.png"
    if not os.path.exists(image_path_7):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_7}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_path_6, caption="Gambar: Hasil Deteksi dan Analisis", use_column_width=True, width=10)
    with col2:
        st.image(image_path_7, caption="Gambar: Hasil Deteksi dan Analisis", use_column_width=True, width=10)
    st.subheader("7. Mengelola Riwayat Deteksi")
    st.markdown("""
    <p style='font-size: 26px; line-height: 1.5; color: #1a1a1a;'>
    Aplikasi menyediakan fitur untuk melihat dan mengelola riwayat deteksi.
    Pengguna dapat melihat riwayat deteksi sebelumnya dan memiliki opsi untuk menghapus riwayat tersebut jika tidak lagi diperlukan.
    </p>
    """, unsafe_allow_html=True)
    # Memusatkan dan mengecilkan gambar untuk langkah 7 menggunakan st.columns
    image_path_8 = "assets/guide/8.png"
    if not os.path.exists(image_path_7):
        st.error(f"Error: Gambar tidak ditemukan di {image_path_7}")
    col1, col2, col3, col4 = st.columns([1, 4, 1, 1])
    with col2:
        st.image(image_path_8, caption="Gambar: Tombol Riwayat Deteksi dan Hapus Riwayat", use_column_width=True, width=10)

    st.markdown("---")
    # Tombol untuk menuju halaman deteksi
    if st.button("Mulai Deteksi Penyakit 🚀"):
        st.session_state.page = "detection"
        st.experimental_rerun()  # Memaksa Streamlit untuk me-render ulang halaman

# Konfigurasi halaman dengan light mode dan font yang diperbesar (global CSS)
st.set_page_config(
    page_title="Deteksi Penyakit Tanaman Daun Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS STYLING YANG SUDAH DIUPGRADE - TAMPILAN MODERN DAN PREMIUM
st.markdown("""
<style>
/* ========== GLOBAL STYLING ========== */
.stApp {
    background-color: #FFFFFF !important; /* Latar belakang putih */
    color: #1a1a1a !important; /* Warna teks hitam */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Header styling */
header[data-testid="stHeader"] {
    background: #FFFFFF !important; /* Header putih */
    color: #1a1a1a !important; /* Warna teks hitam */
    height: 70px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    border-bottom: 2px solid #87CEEB !important; /* Garis biru langit */
}

/* Main content container */
.main .block-container {
    padding: 2rem 2rem !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    margin: 1rem auto !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    max-width: 1400px !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border-radius: 15px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

/* ========== SIDEBAR STYLING ========== */
section[data-testid="stSidebar"] {
    background-color: #87CEEB !important; /* Sidebar biru langit */
    color: #1a1a1a !important; /* Warna teks hitam */
    border-radius: 0 15px 15px 0 !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.15) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    padding: 2rem 1.5rem !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
}

.stSidebar * {
    color: #1a1a1a !important; /* Warna teks hitam untuk semua elemen di sidebar */
}

.stSidebar .stMarkdown p {
    font-size: 28px !important;
    font-weight: 500 !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    margin-bottom: 1.2rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
}

.stSidebar h1, .stSidebar h2 {
    font-size: 32px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: bold !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    margin-bottom: 1.5rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    margin-top: 2rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
}

.stSidebar h3 {
    font-size: 28px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 600 !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    margin-bottom: 1.5rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    margin-top: 1.5rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
}

/* ========== TYPOGRAPHY - MUCH LARGER ========== */
h1 {
    font-size: 4rem !important;
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 2rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    margin-top: 1rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    text-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

h2 {
    font-size: 2.8rem !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 700 !important;
    margin: 2rem 0 1rem 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    text-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

h3 {
    font-size: 2.2rem !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 650 !important;
    margin: 1.5rem 0 1rem 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
}

/* ========== CONTENT TEXT - EXTRA LARGE ========== */
.stMarkdown p {
    font-size: 28px !important;
    line-height: 1.9 !important; /* Line height dikembalikan ke nilai yang lebih nyaman */
    color: #1a1a1a !important; /* Warna teks hitam */
    margin-bottom: 1.2rem !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    font-weight: 500 !important;
}

.stMarkdown li {
    font-size: 26px !important;
    line-height: 1.8 !important; /* Line height dikembalikan ke nilai yang lebih nyaman */
    margin-bottom: 12px !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
}

.stMarkdown strong, .stMarkdown b {
    font-size: 30px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 800 !important;
}

.stMarkdown em, .stMarkdown i {
    font-size: 26px !important;
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    font-style: italic !important;
}

/* ========== BUTTONS - HIGHLY VISIBLE ========== */
/* Menargetkan tombol utama dengan selektor yang lebih spesifik */
.stButton > button {
    font-size: 52px !important; /* Ukuran font tetap besar */
    font-weight: 800 !important;
    padding: 8px 15px !important; /* Padding tetap sempit */
    background-color: #0B6285 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 20px rgba(11, 98, 133, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    min-height: 80px !important; /* Menyesuaikan min-height */
    width: 100% !important;
    margin: 10px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    line-height: 1.2 !important; /* Menambahkan line-height */
}

.stButton > button:hover {
    background-color: #084B66 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(11, 98, 133, 0.6) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(11, 98, 133, 0.4) !important;
}

/* Download button special styling */
.stDownloadButton > button {
    font-size: 48px !important; /* Ukuran font tetap besar */
    font-weight: 800 !important;
    padding: 6px 12px !important; /* Padding tetap sempit */
    background-color: #0B6285 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 20px rgba(11, 98, 133, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    min-height: 70px !important; /* Menyesuaikan min-height */
    line-height: 1.2 !important; /* Menambahkan line-height */
    margin: 10px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
}

.stDownloadButton > button:hover {
    background-color: #084B66 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(11, 98, 133, 0.6) !important;
}

/* ========== SIDEBAR BUTTONS ========== */
.stSidebar .stButton > button {
    font-size: 40px !important; /* Ukuran font tetap besar */
    font-weight: 800 !important;
    padding: 6px 10px !important; /* Padding tetap sempit */
    background-color: #FFFFFF !important;
    color: #0B6285 !important;
    border: 2px solid rgba(255,255,255,0.8) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    min-height: 60px !important; /* Menyesuaikan min-height */
    width: 100% !important;
    margin: 8px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    line-height: 1.2 !important; /* Menambahkan line-height */
}

.stSidebar .stButton > button:hover {
    background-color: #F0F7FA !important;
    color: #084B66 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
    border-color: #ffffff !important;
}

/* Styling untuk tombol radio di sidebar */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    font-size: 24px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 500 !important;
}

/* Styling untuk label slider di sidebar */
section[data-testid="stSidebar"] .stSlider label p {
    font-size: 24px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 500 !important;
}

/* Styling untuk angka nilai slider (misal: "30") */
section[data-testid="stSidebar"] .stSlider .st-bd .st-be {
    font-size: 40px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 900 !important;
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    text-shadow: 0 0 8px rgba(11, 98, 133, 0.6) !important;
}

/* ========== FORM ELEMENTS - LARGER ========== */
.stSelectbox > div > div {
    font-size: 20px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
    background-color: #ffffff !important;
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 8px !important;
    padding: 10px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
}

.stSelectbox > div > div:focus {
    border-color: #0B6285 !important; /* Fokus biru tua */
    box-shadow: 0 0 10px rgba(11, 98, 133, 0.3) !important;
}

.stSlider > div > div > div {
    font-size: 20px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 600 !important;
}

.stSlider > div > div > div > div {
    font-size: 20px !important;
}

/* Styling for the file uploader label text (now hidden) */
.stFileUploader label {
    font-size: 0 !important; /* Sembunyikan ukuran font */
    color: transparent !important; /* Jadikan teks transparan */
    height: 0 !important; /* Runtuhkan tinggi */
    margin: 0 !important; /* Hapus margin */
    padding: 0 !important; /* Hapus padding */
    display: block !important; /* Pastikan mengambil barisnya sendiri */
    overflow: hidden !important; /* Sembunyikan konten yang meluap */
}

/* Styling for the file uploader box itself (drag and drop area) */
.stFileUploader > div {
    font-size: 24px !important; /* Pertahankan ukuran font untuk teks drag and drop */
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
    background-color: #FFFFFF !important;
    border: 2px dashed #87CEEB !important; /* Border putus-putus biru langit */
    border-radius: 12px !important;
    padding: 20px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    text-align: center !important;
}

.stFileUploader > div:hover {
    background-color: #F0F7FA !important; /* Biru lebih terang saat hover */
    border-color: #0B6285 !important; /* Border biru tua saat hover */
}

.stRadio > div {
    font-size: 20px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
}

.stRadio > div > label {
    font-size: 20px !important;
}

.stTextInput > div > div > input {
    font-size: 20px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    background-color: #ffffff !important;
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 8px !important;
    padding: 10px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
}

.stTextInput > div > div > input:focus {
    border-color: #0B6285 !important; /* Fokus biru tua */
    box-shadow: 0 0 10px rgba(11, 98, 133, 0.3) !important;
}

/* ========== ALERTS - LARGER AND MORE VISIBLE ========== */
.stSuccess > div {
    font-size: 24px !important;
    font-weight: 600 !important;
    background-color: #F0F7FA !important; /* Biru lebih terang untuk sukses */
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 12px !important;
    padding: 16px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    box-shadow: 0 4px 15px rgba(135, 206, 235, 0.2) !important;
}

.stWarning > div {
    font-size: 22px !important;
    font-weight: 600 !important;
    background-color: #FFFFF0 !important; /* Kuning lebih terang untuk peringatan */
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    border: 2px solid #F0E68C !important; /* Border khaki */
    border-radius: 12px !important;
    padding: 16px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    box-shadow: 0 4px 15px rgba(243, 156, 18, 0.2) !important;
}

.stError > div {
    font-size: 22px !important;
    font-weight: 600 !important;
    background-color: #FFF0F5 !important; /* Merah muda untuk error */
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    border: 2px solid #F08080 !important; /* Border merah karang */
    border-radius: 12px !important;
    padding: 16px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    box-shadow: 0 4px 15px rgba(231, 76, 60, 0.2) !important;
}

.stInfo > div {
    font-size: 22px !important;
    font-weight: 600 !important;
    background-color: #F0F7FA !important; /* Biru lebih terang untuk info */
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 12px !important;
    padding: 16px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    box-shadow: 0 4px 15px rgba(135, 206, 235, 0.2) !important;
}

/* ========== CONTAINERS ========== */
.stContainer {
    padding: 20px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    margin: 20px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    background-color: #FFFFFF !important;
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 15px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
}

/* ========== EXPANDER STYLING ========== */
.stExpander {
    margin: 15px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    background-color: #ffffff !important;
    border: 2px solid #87CEEB !important; /* Border biru langit */
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
}

.stExpander > div > div > p {
    font-size: 24px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
    line-height: 1.8 !important; /* Line height dikembalikan ke nilai yang lebih nyaman */
}

.stExpander .stMarkdown p {
    font-size: 24px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 500 !important;
}

/* ========== SPECIAL STYLING FOR SUBHEADERS ========== */
.stMarkdown h3, .stMarkdown h4 {
    font-size: 26px !important;
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    font-weight: 700 !important;
    margin: 20px 0 16px 0 !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    padding: 12px 0 !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    border-bottom: 2px solid #1a1a1a !important; /* Border diubah menjadi hitam */
    padding-left: 10px !important;
    border-radius: 8px !important;
}

/* ========== SPINNER AND LOADING ========== */
.stSpinner > div {
    font-size: 22px !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    font-weight: 600 !important;
}

/* ========== IMAGE CAPTIONS ========== */
/* Menargetkan div yang membungkus gambar dan caption */
/* Ini akan memusatkan caption yang dibuat oleh st.image */
.stImage > div > p {
    text-align: center !important;
    font-size: 20px !important;
    color: #555 !important;
    margin-top: 8px !important;
}

/* ========== COLUMN STYLING ========== */
div[data-testid="column"] {
    padding: 16px !important; /* Padding dikembalikan ke nilai yang lebih nyaman */
    margin: 8px 4px !important; /* Margin dikembalikan ke nilai yang lebih nyaman */
    background-color: #ffffff !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
}

/* ========== WEBCAM SPECIFIC STYLING ========== */
div[data-testid="stVideo"] {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
    border: 3px solid #0B6285 !important; /* Border biru tua */
}

/* ========== RESPONSIVE IMPROVEMENTS ========== */
@media (max-width: 768px) {
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        margin-top: 0.8rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin: 1.5rem 0 0.8rem 0 !important;
    }
    
    .stMarkdown p {
        font-size: 22px !important;
        line-height: 1.5 !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Tombol responsif */
    .stButton > button,
    .stDownloadButton > button,
    .stSidebar .stButton > button {
        font-size: 20px !important;
        padding: 10px 15px !important; /* Padding responsif dikembalikan */
        min-height: 60px !important; /* Menyesuaikan tinggi minimum responsif */
        margin: 8px 0 !important;
    }
    
    .main .block-container {
        padding: 1.5rem 1rem !important; /* Padding responsif dikembalikan */
        margin: 0.5rem auto !important; /* Margin responsif dikembalikan */
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem !important; /* Padding sidebar responsif dikembalikan */
    }

    .stSidebar .stMarkdown p,
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        margin-bottom: 0.8rem !important;
        margin-top: 1rem !important;
    }
    
    .stMarkdown li {
        font-size: 20px !important;
        line-height: 1.5 !important;
        margin-bottom: 8px !important;
    }

    .stContainer {
        padding: 15px !important;
        margin: 10px 0 !important;
    }

    .stExpander {
        margin: 10px 0 !important;
    }

    .stMarkdown h3, .stMarkdown h4 {
        margin: 15px 0 10px 0 !important;
        padding: 10px 0 !important;
    }

    .stImage > div {
        margin-top: 5px !important;
    }

    div[data-testid="column"] {
        padding: 12px !important;
        margin: 5px 2px !important;
    }
}

/* ========== ANIMATION ENHANCEMENTS ========== */
.stApp * {
    transition: all 0.3s ease !important;
}

.stContainer:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(0,0,0,0.15) !important;
}

/* ========== CUSTOM SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #0B6285; /* Scrollbar biru tua */
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #084B66; /* Biru tua lebih gelap saat hover */
}

/* Styling untuk label slider dan radio di sidebar */
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stRadio label p {
    font-size: 24px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 500 !important;
    color: #1a1a1a !important; /* Warna teks hitam */
    text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; /* Tambahkan bayangan teks untuk visibilitas */
}

/* Untuk memastikan label radio button itu sendiri juga besar */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    font-size: 24px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 500 !important;
}

/* Styling untuk angka nilai slider (misal: "30") */
section[data-testid="stSidebar"] .stSlider .st-bd .st-be {
    font-size: 40px !important; /* Dikembalikan ke ukuran sebelumnya */
    font-weight: 900 !important;
    color: #1a1a1a !important; /* Diubah menjadi hitam */
    text-shadow: 0 0 8px rgba(11, 98, 133, 0.6) !important; /* Bayangan biru tua */
}


</style>
""", unsafe_allow_html=True)


# Logika utama aplikasi untuk mengelola halaman
if __name__ == "__main__":
    # Inisialisasi state halaman jika belum ada
    if 'page' not in st.session_state:
        st.session_state.page = "homepage"

    # Tampilkan halaman yang sesuai
    if st.session_state.page == "homepage":
        homepage()
    elif st.session_state.page == "detection":
        detection_page()

    # Sidebar untuk navigasi antar halaman
    st.sidebar.markdown("---")
    st.sidebar.header("Pilih Halaman")
    if st.sidebar.button("🏠 Halaman Utama"):
        st.session_state.page = "homepage"
        st.experimental_rerun()
    if st.sidebar.button("🔍 Halaman Deteksi"):
        st.session_state.page = "detection"
        st.experimental_rerun()
