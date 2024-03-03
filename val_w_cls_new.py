import torch
from PIL import Image
import numpy as np
import yaml
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from glob import glob
import os
import argparse
from models.common import DetectMultiBackend
import cv2
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fire/Smoke Detection Workflow')
parser.add_argument('--d-weights', required=True, help='Path to YOLOv5 weights')
parser.add_argument('--c-weights', required=True, help='Path to classification model')
parser.add_argument('--name', required=True, help='Name of result folder')

args = parser.parse_args()

# Load the models
yolov5_model = YOLO(model=args.d_weights)
c_model = DetectMultiBackend(args.c_weights, device=torch.device("cuda:0"))

# Load the YAML file
with open('data/Real.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Get the paths to the images
image_paths = glob(os.path.join(".\dataset\Real", data['test'], '*.jpg'))

# Initialize lists to store the true labels and predicted labels
y_true = []
y_pred = []

# Delete the result directory if it exists, then create it
result_dir = f'runs/detect/{args.name}'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
os.makedirs(os.path.join(result_dir, 'tp'))
os.makedirs(os.path.join(result_dir, 'tn'))
os.makedirs(os.path.join(result_dir, 'fp'))
os.makedirs(os.path.join(result_dir, 'fn'))

def process_image(image_path):
    # Load image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Perform object detection
    results = yolov5_model(img, verbose=False)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu()

    label_path = image_path.replace('.jpg', '.txt').replace('images', 'labels')
    contains_annotated_objects = os.path.exists(label_path) and os.path.getsize(label_path) > 0
    y_true.append(1 if contains_annotated_objects else 0)
    predictions = set()

    for box in boxes:
        # Crop the detected object from the image using the bounding box coordinates
        x1, y1, x2, y2 = box
        obj = img_np[int(y1):int(y2), int(x1):int(x2)]

        # Prepare the object for classification
        obj = torch.from_numpy(obj).permute(2, 0, 1).float().to(torch.device("cuda:0")) / 255.0  # Normalize to [0, 1]
        obj = obj.unsqueeze(0)  # Add batch dimension

        # Perform classification
        output = c_model(obj)
        prediction = output.argmax(dim=1).item()
        predictions.add(prediction)

        # Draw the bounding box on the image
        if prediction in {0, 1}:
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Add the predicted label to the list
    if contains_annotated_objects and predictions.intersection({0, 1}):  # True positive
        case = 'tp'
    elif not contains_annotated_objects and not predictions.intersection({0, 1}):  # True negative
        case = 'tn'
    elif not contains_annotated_objects and predictions.intersection({0, 1}):  # False positive
        case = 'fp'
    else:  # False negative
        case = 'fn'

    # Add the predicted label to the list
    y_pred.append(1 if predictions.intersection({0, 1}) else 0)  # Assuming that classes 0 and 1 are fire/smoke
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # Save the image in the corresponding directory
    result_path = os.path.join(result_dir, case, os.path.basename(image_path))
    cv2.imwrite(result_path, img_np)


# Test the workflow
for image_path in image_paths:
    process_image(image_path)

# Calculate precision, recall, and confusion matrix
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_mat.ravel()
# Calculate False Alarm Rate (FAR)
FAR = fp / (fp + tn)

# Calculate False Negative Rate (FNR)
FNR = fn / (fn + tp)

# Calculate Detection Rate (DR)
DR = tp / (tp + fn)

print(f'Precision: {precision}, Recall: {recall}')
print(f'Confusion Matrix:\n{conf_mat}')
print(f"False Alarm Rate (FAR): {FAR}")
print(f"False Negative Rate (FNR): {FNR}")
print(f"Detection Rate (DR): {DR}")


with open(f'{result_dir}/precision_recall.txt', 'w') as f:
    f.write(f'Precision: {precision}\nRecall: {recall}\nFAR: {FAR}\n FNR: {FNR}\DR: {DR}')


# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f"{result_dir}/confusion_matrix.jpg")
