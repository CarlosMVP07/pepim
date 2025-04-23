from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_type, face_db=None):
        """Inizializza il modello YOLO e carica il database dei volti."""
        self.model = YOLO(model_type)  # Carica la versione specifica di YOLO
        self.face_db = face_db if face_db else {}

    def detect_and_annotate(self, frame):
        """Rileva oggetti nel frame e disegna i bounding boxes."""
        try:
            # Esegui il rilevamento
            results = self.model(frame)
            annotated_frame = frame.copy()

            # Verifica se ci sono rilevamenti
            if results and len(results[0].boxes) > 0:
                print(f"Rilevati {len(results[0].boxes)} oggetti")

                # Itera sui rilevamenti
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinate del bounding box
                    conf = box.conf[0]  # Confidenza
                    cls = int(box.cls[0])  # Classe
                    label = self.model.names.get(cls, "Unknown")  # Nome della classe

                    # Disegna il bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            else:
                print("Nessun oggetto rilevato nel frame.")

            return annotated_frame

        except Exception as e:
            print(f"Errore durante il rilevamento e annotazione: {e}")
            return frame