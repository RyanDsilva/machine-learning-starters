import speech_recognition as sr

AUDIO_FILE = "./test.flac"

rec = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak into th microphone now...")
    audio = rec.listen(source)

# with sr.AudioFile(AUDIO_FILE) as source:
#  audio = rec.record(source)

try:
    print(rec.recognize_wit(audio, key="WIT_AI_KEY"))
except sr.UnknownValueError:
    print("Could not understand audio!")
except sr.RequestError as e:
    print("Could Not connect to server!; {0}".format(e))
