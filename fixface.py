### keeps the face in a move at a fixed point

import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Selfie Segmentation Initialisierung
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Video öffnen
cap = cv2.VideoCapture('input.mp4')

# Zielhintergrund (einfarbig oder ein Bild)
background_image = cv2.imread('background.jpg')  # Lade ein Bild als Hintergrund (z.B. background.jpg)
background_color = (0, 255, 0)  # Grün als Hintergrundfarbe

# Hole die Videoauflösung
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialisiere VideoWriter für das Ausgabevideo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_replaced_bg.mp4', fourcc, fps, (frame_width, frame_height))

# Mediapipe Selfie Segmentation verwenden
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertiere das Bild in RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Selfie-Segmentation ausführen
        results = selfie_segmentation.process(image_rgb)

        # Hole die Segmentierungsmaske
        mask = results.segmentation_mask

        # Schwelle für die Maske (hier kannst du den Wert anpassen, z.B. 0.6 oder 0.7)
        threshold = 0.6
        condition = mask > threshold

        # Maske glätten mit Weichzeichner (um harte Kanten zu reduzieren)
        condition = cv2.GaussianBlur(condition.astype(np.float32), (15, 15), 0)

        # Wenn der Hintergrund ein Bild ist, passe die Größe des Bildes an das Video an
        if background_image is not None:
            bg_resized = cv2.resize(background_image, (frame_width, frame_height))
        else:
            # Erstelle einen einfarbigen Hintergrund
            bg_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            bg_resized[:] = background_color

        # Ersetze den Hintergrund
        condition = condition[..., None]  # Dimension erweitern
        output_frame = np.where(condition > 0.5, frame, bg_resized)

        # Schreibe das Frame in die Ausgabedatei
        out.write(output_frame)

        # Optional: Zeige das aktuelle Frame
        cv2.imshow('Replaced Background Video', output_frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Beendet mit ESC
            break

# Freigeben der Ressourcen
cap.release()
out.release()
cv2.destroyAllWindows()
