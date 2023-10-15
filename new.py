import cv2
import pyttsx3
import numpy as np

class Label:
    def __init__(self, description, confidence):
        self.description = description
        self.confidence = confidence

def describe_image(image_path, net, classes):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    labels = []
    for i in indices:
        #i = i[0]  # Extract the index from the NumPy array
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        labels.append(Label(label, confidence))

    return labels

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def checkRoad(labels, engine):
    road_objects = ["highway", "lane", "road", "car", "motor vehicle", "bicycle", "truck", "traffic"]

    road_count = sum(1 for label in labels if label.description.lower() in road_objects)
    car_count = sum(1 for label in labels if label.description.lower() == "car")

    if road_count >= 1:
        if car_count >= 1:
            engine.say("It seems you are walking on a road with vehicles. Beware! Do you want me to find people for help?")
        else:
            engine.say("It seems the road you are walking on is quite safe. Yet beware.")
    else:
        engine.say("I couldn't identify a road. Please be cautious of your surroundings.")
    engine.runAndWait()

def main(image_path, net, classes):
    labels = describe_image(image_path, net, classes)
    print("Image Description:", [label.description for label in labels])
    engine = pyttsx3.init()
    engine.say("I see " + ", ".join([label.description for label in labels]))
    engine.runAndWait()
    checkRoad(labels, engine)

if __name__ == "__main__":
    net = cv2.dnn.readNet(r"C:\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\yolo\yolov3.weights", r"C:\\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\yolo\yolov3.cfg")
    with open("yolo/coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    image_path = r"images\road.jpg"
    main(image_path, net, classes)
