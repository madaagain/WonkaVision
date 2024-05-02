import cv2
import os
from ultralytics import YOLO


IMAGE_DIR = os.path.join('.', 'videos')

image_path = os.path.join(IMAGE_DIR, 'passionfruit.jpg')
image_path_out = '{}_out.png'.format(image_path)
model_path = os.path.join('.', 'runs', 'detect', 'train16', 'weights', 'last.pt')

# Charger le modèle YOLO
model = YOLO(model_path)

# Charger l'image
image = cv2.imread(image_path)

# Vérifier si l'image a été chargée correctement
if image is None:
    print("Erreur : Impossible de charger l'image.")
    exit()

# Appliquer le modèle à l'image
results = model(image)[0]

# Seuil de confiance
threshold = 0.003

# Dessiner les rectangles de détection
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, results.names[int(class_id)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Afficher l'image
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
