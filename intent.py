
import os 
import dialogflow as dialogflow
import speech_recognition as sr
from gtts import gTTS
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
