import cv2
import time
import numpy as np

class YoloDetector:
    def __init__(self, labelsPath, weightsPath, configPath):
        self.LABELS = open(labelsPath).read().strip().split("\n")
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detectYolo(self, frame, args):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        return idxs, boxes, classIDs, confidences

    def detectAndShow(self, frame, args):
        idxs, boxes, classIDs, confidences = self.detectYolo(frame, args)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                color = (0, 0, 0)  # Set color to black (RGB: 0, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print("Number of detected objects:", len(idxs))
        print("Detected class IDs:", classIDs)
        print("Detected confidences:", confidences)
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
    

def main():
    labelsPath = "yolo/coco.names"
    weightsPath = "yolo/yolov3.weights"
    configPath = "yolo/yolov3.cfg"
    args = {"confidence": 0.5, "threshold": 0.3}

    yolo_detector = YoloDetector(labelsPath, weightsPath, configPath)
    frame = cv2.imread(r"C:\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\images/road.jpg")  # Replace this with the actual path to your input image
    if frame is None:
        print("Error: Unable to load the image.")
    else:
        print("Image loaded successfully.")
        yolo_detector.detectAndShow(frame, args)

if __name__ == "__main__":
    main()
