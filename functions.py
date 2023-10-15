import cv2
import numpy as np
import wave
import pyaudio



def getBrightness(cam):
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = np.sum(frame)/(frame.shape[0]*frame.shape[1])
    avg=avg/255
    if(avg > 0.6):
        return ("Very bright", avg)
    if(avg > 0.4):
        return ("Bright", avg)
    if(avg>0.2):
        return ("Dim", avg)
    else:
        return ("Dark",avg)


def play_file(fname):
    # create an audio object
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()
    chunk = 1024

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != '':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

        # cleanup stuff.
    stream.close()
    p.terminate()

# Define your functions (getBrightness and play_file) here

# Assuming your play_file and getBrightness functions are defined above this point in your script

# Example usage of the play_file function
#if __name__ == "__main__":
    #audio_file_path = 'ping.wav'  # If the audio file is in the same directory as your script
    #play_file(audio_file_path)

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)  # Open the default camera (index 0)
    audio_file_path = 'ping.wav'  # If the audio file is in the same directory as your script
    
    # Call getBrightness function passing the cam object
    brightness_label, avg_brightness = getBrightness(cam)
    
    # Print the brightness label and average brightness
    print("Brightness Label:", brightness_label)
    print("Average Brightness:", avg_brightness)

    # Call play_file function passing the audio file path
    play_file(audio_file_path)

    # Release the camera
    cam.release()




