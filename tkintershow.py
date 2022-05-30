import tkinter
import cv2
import PIL.Image
import PIL.ImageTk

'''
#
# Readme in case malgitnich
#
# diri model.predict ta3ak hadik nrml
# w nrmlmn fi code ta3 dek li ykhabrk chtaho emotion rahi kayn win ydir print wla haja ta3 dek prediction
# aya nti jibi hadak prediction w dirilah str() w 7otih fi text li rahi tahta
# w bedli path li rah tehta diri ta3 image source w bedli path tani ta3 haarcascade hadak hh
# lsl diri bikhtisar swalhk nti raki fahmtni jc wch rani ngol hhhh 
# aya w see'ya w kiss your mom for me hh 
#
'''

text = 'Hello World'  # text li tjibih mn prediction dirih hnaya

# Create a window
window = tkinter.Tk()
window.title("OpenCV and Tkinter")

# Load an image using OpenCV
# path ta3 img li rayha tkhdmi 3liha
path = 'dataset/IMG_20220429_231212_435.jpg'
faceCascade = cv2.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml')  # path ta3 haarcascade ta3ak

frame = cv2.imread(path)
faces = faceCascade.detectMultiScale(
    frame,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30))


(x, y, w, h) = faces[0]
# the frame that contains the image with square
frame = frame[y:y+h, x:x+w]
frame = cv2.resize(frame, (640, 480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, str(text), (x+5, y-5),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# bg bedli fiha bkach color y3jbk
canvas2 = tkinter.Canvas(window, width=800, height=480, bg="SpringGreen2")

# Add a text in Canvas
canvas2.create_text(300, 50, text=text,
                    fill="black", font=('Helvetica 15 bold'))
canvas2.grid(column=2, row=0)

# Create a canvas that contains above image
canvas = tkinter.Canvas(window, width=800, height=480)
canvas.grid(column=0, row=0)


# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

# Run the window loop
window.mainloop()
