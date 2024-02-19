import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
import tkinter as tk
from tkinter import filedialog, Label, Canvas, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    return warped
  def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        perform_document_scanning(file_path)



def perform_document_scanning(image_path):
    original_img = cv2.imread(image_path)
    copy = original_img.copy()
    plt.imshow(copy, cmap='gray')
    plt.title('Original Image')
    plt.show()

    ratio = original_img.shape[0] / 500.0
    img_resize = imutils.resize(original_img, height=500)

    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gray Image')
    plt.show()

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    plt.imshow(blurred_image, cmap='gray')
    plt.title('Blurred Image')
    plt.show()

    edged_img = cv2.Canny(blurred_image, 75, 200)
    plt.imshow(edged_img, cmap='gray')
    plt.title('Edge Image')
    plt.show()

    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) >= 4:
            doc = approx
            break

    p = []

    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)

    x, y = zip(*p)
    plt.scatter(x, y, c='r', marker='o')
    plt.imshow(img_resize)
    plt.title('Circled Corner Points')
    plt.show()

    warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
    cv2.imshow("Scanned Image", imutils.resize(warped_image, height=650))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(0)


# Tkinter setup
root = tk.Tk()
root.title("Document Scanner")

process_button = tk.Button(root, text="Select Image", command=open_file_dialog)
canvas = Canvas(root, width=620, height=300)

# Place GUI elements on the window
process_button.pack(pady=10)
canvas.pack()

# Run the Tkinter event loop
root.mainloop()
