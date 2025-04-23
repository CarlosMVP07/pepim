import sys
import cv2
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QWidget, QProgressBar, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from detector_v2 import YOLOFaceDetector
from utils_v2 import load_face_db

# Configurazioni generali
FACE_DB_DIR = "faces"
URL_HISTORY_FILE = "url_history.json"

# Mappa delle versioni YOLO ai modelli disponibili
YOLO_MODELS = {
    "YOLOv5": ["YOLOv5n", "YOLOv5s", "YOLOv5m", "YOLOv5l", "YOLOv5x"],
    "YOLOv6": ["YOLOv6n", "YOLOv6s", "YOLOv6m", "YOLOv6l", "YOLOv6x"],
}

class VideoProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Face and Object Detection")
        self.setGeometry(100, 100, 800, 600)

        # Layout principale
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Etichette e bottoni
        self.video_label = QLabel("Carica un video o inserisci un URL per iniziare")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.upload_button = QPushButton("Carica Video Locale")
        self.upload_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.upload_button)

        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("Inserisci l'URL per il Live Video")
        self.url_input.textChanged.connect(self.on_url_input_change)
        self.layout.addWidget(self.url_input)

        self.url_history = QComboBox()
        self.url_history.currentTextChanged.connect(self.load_from_history)
        self.layout.addWidget(self.url_history)

        # Selettore versione YOLO
        self.yolo_version_selector = QComboBox()
        self.yolo_version_selector.addItems(YOLO_MODELS.keys())
        self.yolo_version_selector.currentIndexChanged.connect(self.populate_model_selector)
        self.layout.addWidget(self.yolo_version_selector)

        # Selettore modello YOLO
        self.yolo_model_selector = QComboBox()
        self.layout.addWidget(self.yolo_model_selector)

        self.process_button = QPushButton("Elabora Video/Live Stream")
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        # Variabili di stato
        self.video_path = None
        self.live_stream_url = None
        self.detector = None
        self.selected_yolo_version = None
        self.selected_yolo_model = None

        # Inizializza il selettore di modelli
        self.populate_model_selector()

        # Inizializza il modello e il database facciale
        self.init_model()
        self.load_url_history()

    def init_model(self):
        """Carica il modello YOLO selezionato."""
        if not self.selected_yolo_model:
            self.video_label.setText("Errore: Nessun modello YOLO selezionato.")
            return

        model_path = f"models/{self.selected_yolo_model.lower()}.pt"
        try:
            self.face_db = load_face_db(FACE_DB_DIR)
            self.detector = YOLOFaceDetector(model_type=model_path, face_db=self.face_db)
            self.video_label.setText(f"Caricato modello {self.selected_yolo_model} con successo!")
        except Exception as e:
            self.detector = None
            self.video_label.setText(f"Errore durante il caricamento del modello: {e}")
            print(f"Errore caricamento modello: {e}")

    def populate_model_selector(self):
        """Popola il selettore di modelli in base alla versione YOLO selezionata."""
        self.selected_yolo_version = self.yolo_version_selector.currentText()
        self.yolo_model_selector.clear()
        self.yolo_model_selector.addItems(YOLO_MODELS[self.selected_yolo_version])
        self.selected_yolo_model = self.yolo_model_selector.currentText()
        self.yolo_model_selector.currentIndexChanged.connect(self.update_selected_model)

    def update_selected_model(self):
        """Aggiorna il modello YOLO selezionato."""
        self.selected_yolo_model = self.yolo_model_selector.currentText()
        self.init_model()

    def load_video(self):
        """Carica un video locale tramite finestra di dialogo."""
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(
            self, "Seleziona un Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if self.video_path:
            self.video_label.setText(f"Video selezionato: {self.video_path}")
            self.process_button.setEnabled(True)
            self.url_input.setEnabled(False)

    def on_url_input_change(self):
        """Abilita/disabilita pulsanti in base all'input URL."""
        if self.url_input.text():
            self.upload_button.setEnabled(False)
            self.process_button.setEnabled(True)
        else:
            self.upload_button.setEnabled(True)
            self.process_button.setEnabled(False)

    def load_from_history(self, url):
        """Carica un URL dalla cronologia."""
        if url:
            self.url_input.setText(url)
            self.process_button.setEnabled(True)

    def process_video(self):
        """Elabora il video o il flusso live e mostra il progresso."""
        if self.url_input.text():
            self.live_stream_url = self.url_input.text()
            self.video_label.setText(f"Elaborazione del flusso live: {self.live_stream_url}")
            self.save_url_to_history(self.live_stream_url)
            video_source = self.live_stream_url
        elif self.video_path:
            self.video_label.setText(f"Elaborazione del video: {self.video_path}")
            video_source = self.video_path
        else:
            self.video_label.setText("Errore: Nessun video o URL selezionato.")
            return

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.video_label.setText(f"Errore: impossibile aprire il video o flusso {video_source}")
            print(f"Errore apertura video: {video_source}")
            return

        # Leggi i frame dal flusso
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore durante la lettura del frame.")
                self.video_label.setText("Errore: impossibile leggere il frame dal flusso video.")
                break

            # Rilevamento e annotazione
            if self.detector:
                frame = self.detector.detect_and_annotate(frame)

            # Converti il frame in un formato compatibile con PyQt5
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                frame_rgb.strides[0], QImage.Format_RGB888
            )

            # Mostra il frame nell'etichetta
            self.video_label.setPixmap(QPixmap.fromImage(qimage).scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
            ))

            # Interrompi il ciclo se la finestra viene chiusa
            QApplication.processEvents()

        cap.release()

    def save_url_to_history(self, url):
        """Salva l'URL nella cronologia."""
        try:
            with open(URL_HISTORY_FILE, "r") as file:
                history = json.load(file)
        except FileNotFoundError:
            history = []

        if url not in history:
            history.append(url)
            with open(URL_HISTORY_FILE, "w") as file:
                json.dump(history, file)

        self.load_url_history()

    def load_url_history(self):
        """Carica la cronologia degli URL salvati."""
        try:
            with open(URL_HISTORY_FILE, "r") as file:
                history = json.load(file)
        except FileNotFoundError:
            history = []

        self.url_history.clear()
        self.url_history.addItems(history)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessorApp()
    window.show()
    sys.exit(app.exec_())