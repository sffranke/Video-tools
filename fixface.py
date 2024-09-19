### keeps the face in a move at a fixed point
### ffmpeg -i input.mp4 -b:v 800k -bufsize 800k -maxrate 800k -b:a 128k output.mp4

import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Initialisierung
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Video öffnen
cap = cv2.VideoCapture('input-800.mp4')

# Zielposition für das Gesicht (z.B. die Mitte des Bildschirms)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate des Videos
target_x, target_y = frame_width // 2, frame_height // 2  # Zielposition in der Mitte des Bildschirms

# VideoWriter initialisieren (für MP4-Ausgabe)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für MP4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Glättungsfaktoren initialisieren
smooth_dx, smooth_dy = 0, 0
alpha = 0.2  # Glättungsfaktor (zwischen 0 und 1; kleiner bedeutet mehr Glättung)
min_movement_threshold = 5  # Minimaler Bewegungsschwellenwert, um kleine Verschiebungen zu ignorieren
max_shift_per_frame = 303    # Maximale Bewegung pro Frame, um abrupte Bewegungen zu vermeiden
#stabilization_reset_interval = 300  # Nach dieser Anzahl von Frames wird der Versatz hart zurückgesetzt

# Frame-Zähler für das Zurücksetzen
frame_counter = 0

# Mediapipe Face Detection initialisieren
with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

    first_face_detected = False  # Flag, ob das erste Gesicht erkannt wurde

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere das Bild in RGB (Mediapipe erwartet RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Gesichtserkennung
        results = face_detection.process(image_rgb)

        # Wenn ein Gesicht erkannt wurde
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x_center = int((bboxC.xmin + bboxC.width / 2) * iw)
                y_center = int((bboxC.ymin + bboxC.height / 2) * ih)

                # Wenn das erste Gesicht erkannt wurde, speichere seine Position
                if not first_face_detected:
                    first_face_detected = True
                    print(f"Erstes Gesicht erkannt bei: ({x_center}, {y_center})")

                # Berechne den Versatz, um das Gesicht in der Mitte des Bildes zu halten
                dx = target_x - x_center
                dy = target_y - y_center

                # Nur verschieben, wenn die Bewegung größer als der Schwellenwert ist
                if abs(dx) > min_movement_threshold or abs(dy) > min_movement_threshold:
                    # Glättung der Bewegung durch einen gleitenden Durchschnitt
                    smooth_dx = int(alpha * dx + (1 - alpha) * smooth_dx)
                    smooth_dy = int(alpha * dy + (1 - alpha) * smooth_dy)

                    # Begrenze die maximale Bewegung pro Frame
                    smooth_dx = np.clip(smooth_dx, -max_shift_per_frame, max_shift_per_frame)
                    smooth_dy = np.clip(smooth_dy, -max_shift_per_frame, max_shift_per_frame)

                    # Verschiebe das Bild, um das Gesicht zu zentrieren
                    M = np.float32([[1, 0, smooth_dx], [0, 1, smooth_dy]])
                    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                break  # Nur das erste erkannte Gesicht verwenden

        # Frame-Zähler für das Zurücksetzen
        '''
        frame_counter += 1
        if frame_counter % stabilization_reset_interval == 0:
            # Harte Stabilisierung nach einer bestimmten Anzahl Frames
            print("Harte Stabilisierung. Setze Versatz zurück.")
            smooth_dx, smooth_dy = 0, 0
        '''
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
