import cv2
import pytesseract
from gtts import gTTS
import os 
from google.cloud import dialogflow_v2 as dialogflow  







import speech_recognition as sr
from gtts import gTTS
from google.protobuf.json_format import MessageToJson

# Load the image
image = cv2.imread('bank.jpeg')  # Replace 'your_image.jpg' with the image file's path

# Perform OCR to extract text from the image
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
tts = gTTS(text)

# Save the speech to an audio file
tts.save("output.mp3")

# Play the generated speech using your system's default audio player
import os
os.system("start output.mp3")  # On Windows

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
    