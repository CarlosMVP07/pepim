from ultralytics import YOLO
import cv2

def detect_objects(video_path, model_path="runs/train/exp/weights/best.pt"):
    """
    Rileva oggetti in un video utilizzando un modello YOLO addestrato.

    :param video_path: Percorso del file video.
    :param model_path: Percorso del modello YOLO addestrato.
    """
    # Carica il modello YOLO addestrato
    model = YOLO(model_path)

    # Apri il video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video {video_path}")
        return

    # Leggi i frame dal video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rileva oggetti nel frame
        results = model(frame)

        # Annotazioni sui frame (bounding boxes)
        annotated_frame = results[0].plot()

        # Mostra il frame annotato
        cv2.imshow("Rilevamento Oggetti", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Specifica il percorso del tuo video
    detect_objects(video_path)