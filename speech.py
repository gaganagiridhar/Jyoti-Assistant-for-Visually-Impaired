import speech_recognition as sr
import pyttsx3
from google.oauth2 import service_account

class speech_to_text:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        en_voice_id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"
        self.engine.setProperty('voice', en_voice_id)
        self.credentials = service_account.Credentials.from_service_account_file('C:\\Users\\DELL\\Desktop\\mlpro\\Assistant-for-visually-impaired-master\\machinelearning-401311-b5e6d1706c38.json')

    def recognize_speech_from_mic(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
        print("Listening...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            print("Audio captured:", audio)
        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
            print("Transcription successful:", response["transcription"])
        except sr.RequestError:
            response["success"] = False
            response["error"] = "API unavailable"
            print("API unavailable")
        except sr.UnknownValueError:
            response["error"] = "Unable to recognize speech"
            print("Unable to recognize speech")
        if response["transcription"] is None:
            print("Speech not detected! Please try again!")
        else:
            print("Transcribed Text:", response["transcription"])
        return response["transcription"]

# Example usage
if __name__ == "__main__":
    speech_recognizer = speech_to_text()
    transcribed_text = speech_recognizer.recognize_speech_from_mic()
  
