import cv2
import numpy as np
import wave
import pyaudio
import speech_recognition as sr
from gtts import gTTS
import os
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow
import pyttsx3
import pytesseract
import time




 
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
    
    
       

    def getBrightness(cam):
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg = np.sum(frame) / (frame.shape[0] * frame.shape[1])
        avg = avg / 255
        if avg > 0.6:
            return "Very bright", avg
        elif avg > 0.4:
            return "Bright", avg
        elif avg > 0.2:
            return "Dim", avg
        else:
            return "Dark", avg
    
    def play_file(fname):
        wf = wave.open(fname, 'rb')
        p = pyaudio.PyAudio()
        chunk = 1024
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(chunk)

        while data != b'':
            stream.write(data)
            data = wf.readframes(chunk)

        stream.close()
        p.terminate()

   
    
class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def recognize_speech_from_mic(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source)
            print("Audio captured:", audio)
        
        try:
            transcription = self.recognizer.recognize_google(audio)
            print("Transcription successful:", transcription)
            return transcription
        except sr.RequestError:
            print("API unavailable")
            return None
        except sr.UnknownValueError:
            print("Unable to recognize speech")
            return None
        
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
    


intents = {
    "greeting": ["hello", "hi", "hey"],
    "farewell": ["goodbye", "bye", "see you"],
    "question": ["what", "how", "why", "when"],
    "thanks": ["thank you", "thanks"],
}

# Function to recognize the intent based on input text


def recognize_intent(input_text):
    input_text = input_text.lower()
    for intent, patterns in intents.items():
        for pattern in patterns:
            if pattern in input_text:
                return intent
    return "unknown"  # If no intent is recognized

# Initialize the recognizer
recognizer = sr.Recognizer()

# Capture user speech input
with sr.Microphone() as source:
    print("Listening...")
    audio = recognizer.listen(source)

try:
    # Recognize speech input
    user_input = recognizer.recognize_google(audio).lower()
    print("You said:", user_input)

    # Recognize the intent
    recognized_intent = recognize_intent(user_input)

    # Handle the recognized intent and convert the responses to speech
    if recognized_intent == "greeting":
        response = "Hello! How can I help you?"
    elif recognized_intent == "How are you":
        response = "I'm good! Have a great day!"
    elif recognized_intent == "question":
        response = "I'm here to answer your questions."
    elif recognized_intent == "thanks":
        response = "You're welcome!"
    #else:
        #response = "I'm not sure how to respond to that."

    # Convert the response to speech
    tts = gTTS(response, lang='en')
    tts.save("response.mp3")

    # Play the generated speech using your system's default audio player
    os.system("start response.mp3")  # On Windows

except sr.UnknownValueError:
    print("Sorry, I couldn't understand the audio.")
except sr.RequestError as e:
    print("Could not request results; check your network connection.")
    
    


    

    # Recognize speech from the microphone
    


# Optional: Print the extracted text
def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        # print('=' * 20)
        # print('Query text: {}'.format(response.query_result.query_text))
        # print('Detected intent: {} (confidence: {})\n'.format(
        #     response.query_result.intent.display_name,
        #     response.query_result.intent_detection_confidence))
        # print('Fulfillment text: {}\n'.format(
        #     response.query_result.fulfillment_text))
    return response.query_result.intent.display_name, response.query_result.fulfillment_text

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

if __name__ == "__main__":
    labelsPath = "yolo/coco.names"
    weightsPath = "yolo/yolov3.weights"
    configPath = "yolo/yolov3.cfg"
    args = {"confidence": 0.5, "threshold": 0.3}
    speech_recognizer = SpeechToText()
    transcribed_text = speech_recognizer.recognize_speech_from_mic()
    if transcribed_text:
        print("Transcribed Text:", transcribed_text)

        

    speech_to_text = SpeechToText()  
    
    yolo_detector = YoloDetector(labelsPath, weightsPath, configPath)
    frame = cv2.imread(r"C:\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\images\road.jpg")  # Replace this with the actual path to your input image
    if frame is None:
        print("Error: Unable to load the image.")
    else:
        print("Image loaded successfully.")
    yolo_detector.detectAndShow(frame,args)

    # Create an instance of the SpeechToText class
    
    cam = cv2.VideoCapture(0)  # Open the default camera (index 0)
    audio_file_path = 'ping.wav'  # If the audio file is in the same directory as your script

    # Get brightness and play audio
    brightness_label, avg_brightness = YoloDetector.getBrightness(cam)
    print("Brightness Label:", brightness_label)
    print("Average Brightness:", avg_brightness)
    YoloDetector.play_file(audio_file_path)

    # Load and process the image

    # Check brightness after displaying detected objects
    

    # Release the camera
    cam.release()

    # Replace this with the actual path to your input image

    # Detect objects in the image and describe them
    labels = describe_image(r"C:\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\images\road.jpg", yolo_detector.net, yolo_detector.LABELS)
    print("Image Description:", [label.description for label in labels])

    # Process the detected objects and check for road safety
    engine = pyttsx3.init()
    engine.say("I see " + ", ".join([label.description for label in labels]))
    
    engine.runAndWait()
    checkRoad(labels, engine)

    # ... (your existing code for speech recognition, intent recognition, and response generation remains unchanged) ...
image = cv2.imread(r'C:\Users\DELL\Desktop\mlpro\Assistant-for-visually-impaired-master\bank.jpeg') 

text = pytesseract.image_to_string(image)
try:
    # Create a gTTS object to convert text to speech
    tts = gTTS(text, lang='en')
    
    # Save the speech to an audio file
    tts.save("output.mp3")
    
    # Play the generated speech using your system's default audio player
    os.system("start output.mp3")  # On Windows
except Exception as e:
    print("Error:", e)

# Optional: Print the extracted text
print("Extracted Text:", text)

# Create a gTTS object to convert text to speech


# Save the speech to an audio file

# Play the generated speech using your system's default audio player


# Optional: Print the extracted text
def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

    return response.query_result.intent.display_name, response.query_result.fulfillment_text

cv2.waitKey(0)
cv2.destroyAllWindows
   