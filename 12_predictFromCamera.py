import cv2
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/12_cnn.pth"


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()


def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    pts = cv2.findNonZero(thresh)
    if pts is None:
        return None, None

    x, y, w, h = cv2.boundingRect(pts)
    if w * h < 150:
        return None, None

    digit = thresh[y:y + h, x:x + w]

    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = digit

    x = canvas.astype(np.float32) / 255.0
    x = 1.0 - x
    x = (x - 0.5) / 0.5

    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor, canvas


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the camera.")

BOX = 300

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    x1, y1 = (w - BOX) // 2, (h - BOX) // 2
    x2, y2 = x1 + BOX, y1 + BOX

    roi = frame[y1:y2, x1:x2]
    tensor, preview = preprocess(roi)

    label_text = "No digit detected"
    color = (0, 0, 255)

    if tensor is not None:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            conf, pred = probs.max(dim=0)
        label_text = f"Prediction: {pred.item()} ({conf.item() * 100:.1f}%)"
        color = (0, 255, 0)

        prev_big = cv2.resize(preview, (112, 112), interpolation=cv2.INTER_NEAREST)
        frame[10:122, w - 122:w - 10] = cv2.cvtColor(prev_big, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "Quit: q", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("MNIST Real-Time Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()