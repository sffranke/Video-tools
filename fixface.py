### keeps the face in a move at a fixed point
### ffmpeg -i input.mp4 -b:v 800k -bufsize 800k -maxrate 800k -b:a 128k output.mp4
import cv2
import mediapipe as mp
import numpy as np
import argparse

# Mediapipe Initialisierung
mp_face_detection = mp.solutions.face_detection

# Video öffnen
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Der Eingabepfad zur Datei")
args = parser.parse_args()

print(f"Eingabepfad: {args.input}")
print(f"Ausgabepfad: {'centered_' + args.input}")

cap = cv2.VideoCapture(args.input)

# Zielposition für das Gesicht (Mitte des Bildschirms)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate des Videos
target_x, target_y = frame_width // 2, frame_height // 2

# VideoWriter initialisieren (für MP4-Ausgabe)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('centered_' + args.input, fourcc, fps, (frame_width, frame_height))

# Glättungsfaktoren initialisieren
smooth_dx, smooth_dy = 0, 0
alpha = 0.25  # Erhöhter Glättungsfaktor
min_movement_threshold = 5  # Minimaler Bewegungsschwellenwert, um kleine Verschiebungen zu ignorieren
max_shift_per_frame = 150  # Begrenze die maximale Bewegung pro Frame

# Puffer für nicht erkannte Gesichter
last_known_position = None
frames_without_detection = 0
max_frames_without_detection = 10  # Toleranz für fehlende Erkennung

# Mediapipe Face Detection initialisieren
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere das Bild in RGB (Mediapipe erwartet RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gesichtserkennung
        results = face_detection.process(image_rgb)

        if results.detections:
            frames_without_detection = 0  # Gesicht erkannt, Zähler zurücksetzen

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x_center = int((bboxC.xmin + bboxC.width / 2) * iw)
                y_center = int((bboxC.ymin + bboxC.height / 2) * ih)

                # Speichere die letzte bekannte Position
                last_known_position = (x_center, y_center)

                # Berechne den Versatz, um das Gesicht in der Mitte des Bildes zu halten
                dx = target_x - x_center
                dy = target_y - y_center

                # Nur verschieben, wenn die Bewegung größer als der Schwellenwert ist
                if abs(dx) > min_movement_threshold or abs(dy) > min_movement_threshold:
                    smooth_dx = int(alpha * dx + (1 - alpha) * smooth_dx)
                    smooth_dy = int(alpha * dy + (1 - alpha) * smooth_dy)

                    # Begrenze die maximale Bewegung pro Frame
                    smooth_dx = np.clip(smooth_dx, -max_shift_per_frame, max_shift_per_frame)
                    smooth_dy = np.clip(smooth_dy, -max_shift_per_frame, max_shift_per_frame)

                # Verschiebe das Bild, um das Gesicht zu zentrieren
                M = np.float32([[1, 0, smooth_dx], [0, 1, smooth_dy]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                break  # Nur das erste erkannte Gesicht verwenden
        else:
            frames_without_detection += 1

            # Wenn für einige Frames kein Gesicht erkannt wurde, verwende die letzte bekannte Position
            if last_known_position and frames_without_detection <= max_frames_without_detection:
                dx = target_x - last_known_position[0]
                dy = target_y - last_known_position[1]

                smooth_dx = int(alpha * dx + (1 - alpha) * smooth_dx)
                smooth_dy = int(alpha * dy + (1 - alpha) * smooth_dy)

                # Begrenze die maximale Bewegung pro Frame
                smooth_dx = np.clip(smooth_dx, -max_shift_per_frame, max_shift_per_frame)
                smooth_dy = np.clip(smooth_dy, -max_shift_per_frame, max_shift_per_frame)

                # Verschiebe das Bild
                M = np.float32([[1, 0, smooth_dx], [0, 1, smooth_dy]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        # Schreibe den Frame in die Ausgabedatei
        out.write(frame)

        # Zeige das stabilisierte Video an (optional)
        cv2.imshow('Stabilized Video', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Beendet mit ESC
            break

# Freigeben der Ressourcen
cap.release()
out.release()
cv2.destroyAllWindows()
