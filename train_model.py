from ultralytics import YOLO

def train_yolo():
    """
    Addestra un modello YOLO utilizzando il dataset COCO8.
    """
    # Carica il modello YOLO pre-addestrato
    model = YOLO("yolov8n.pt")  # Cambia con il modello desiderato (es. yolov8m.pt, yolov8l.pt)

    # Addestra il modello sul dataset COCO8
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

    # Stampa i risultati dell'addestramento
    print("Addestramento completato!")
    print(results)

if __name__ == "__main__":
    train_yolo()