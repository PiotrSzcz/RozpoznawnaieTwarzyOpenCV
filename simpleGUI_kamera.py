import PySimpleGUI as gui
import cv2 as cv

layout =[[gui.Image(key = '-IMAGE-')],
        [gui.Text('', key='-TEXT-', expand_x=True, justification='c')]]

okno = gui.Window('Rozpoznawanie Twarzy', layout)

face_recognition = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

    event, values = okno.read(timeout=0)
    if event == gui.WIN_CLOSED:
        break

    _, frame = cap.read()
    mono_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    twarze = face_recognition.detectMultiScale(mono_frame, scaleFactor=1.3, minNeighbors=7, minSize=(50,50))

    for (x, y, w, h) in twarze:
        cv.circle(frame, (x+int(w/2), y+int(h/2)), int(w/2), (0,0,255), 1)

    byte_cap = cv.imencode('.png', frame)[1].tobytes()
    okno['-TEXT-'].update(f"Liczba rozpoznanych obiektów: {len(twarze)}")
    okno['-IMAGE-'].update(data = byte_cap)

okno.close()
