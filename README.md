Jyoti: Assistant for the Visually Impaired
Jyoti is a voice assistant specifically designed to assist and empower visually impaired individuals. This system uses speech input and output to provide various features tailored to enhance the lives of the visually impaired.

Key Features:
1. Description:
Environment Description: Jyoti provides a concise, spoken description of the user's surroundings, helping the visually impaired navigate and understand their environment better.

Road Conditions: Jyoti offers insights into road conditions, allowing users to make informed decisions when traveling.

Object and People Recognition: Jyoti identifies and counts objects, people, and more within the webcam's frame, enhancing user awareness.

2. Read:
Text Recognition: Jyoti can detect and read text from images, enabling users to access written information.
3. Fill Forms:
Form Reading: Jyoti reads out forms, particularly useful for handling paperwork related to banking and other applications.
4. Chatbot Features:
Basic Conversations: Jyoti also serves as a chatbot, assisting users with common inquiries, lighting conditions, and other everyday tasks.
Tech Stacks Used:
Speech Recognition: Utilizes Google's API for speech-to-text conversion.

Text-to-Speech: Employs the pyttsx3 Python library for generating spoken responses.

Object Recognition: Leverages the COCO Dataset for object recognition.

Google Cloud Vision API: Used for advanced image processing and text recognition.

Dialogflow: Enables natural language understanding for chatbot interactions.

Installation and Setup:
Install Dependencies.
Download the necessary credentials for the Google Vision API.
Place the credential files in the required locations.
Running the Code:
To run the Jyoti system, execute the following command:

shell
Copy code
python main.py