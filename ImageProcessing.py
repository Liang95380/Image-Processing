# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
from tkinter import filedialog
import transform
import png
import numpy as np


class Image2:
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename=''):
        # you need to input either filename OR x_pixels, y_pixels, and num_channels
        self.input_path = 'input/'
        self.output_path = 'output/'
        if x_pixels and y_pixels and num_channels:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels))
        elif filename:
            self.array = self.read_image(filename)
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
        else:
            raise ValueError("You need to input either a filename OR specify the dimensions of the image")

    def read_image(self, filename, gamma=2.2):
        '''
        read PNG RGB image, return 3D numpy array organized along Y, X, channel
        values are float, gamma is decoded
        '''
        im = png.Reader(self.input_path + filename).asFloat()
        resized_image = np.vstack(list(im[2]))
        resized_image.resize(im[1], im[0], 3)
        resized_image = resized_image ** gamma
        return resized_image

    def write_image(self, output_file_name, gamma=2.2):
        '''
        3D numpy array (Y, X, channel) of values between 0 and 1 -> write to png
        '''
        im = np.clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        im = im.reshape(y, x * 3)
        writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, 255 * (im ** (1 / gamma)))

        self.array.resize(y, x, 3)  # we mutated the method in the first step of the function


class UserGUI:
    def __init__(self):
        self.imgname = 0
        self.im = 0
        self.filter = 0
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = False
        self.blur_isClicked = False
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = False
        self.sobel_xy_isClicked = False
        self.sharpen_isClicked = False


    def select_image(self):
        # grab a reference to the image panels
        global panelA, panelB
        # open a file chooser dialog and allow the user to select an input
        # image

        self.path = filedialog.askopenfilename()
        self.imgname = self.path.split("/")[-1]
        self.im = Image2(filename=self.imgname)
        brightness = transform.brighten(self.im, slider.get())
        brightness.write_image('output.png')
        output_path = 'output/output.png'


        # ensure a file path was selected
        if len(self.path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            image = cv2.imread(self.path)
            filtered_img = cv2.imread(output_path)

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (550, 550))
            filtered_img = cv2.resize(filtered_img, (550, 550))
            # convert the images to PIL format...
            image = Image.fromarray(image)
            filtered_img = Image.fromarray(filtered_img)
            # ...and then to ImageTk format
            image = ImageTk.PhotoImage(image)
            filtered_img = ImageTk.PhotoImage(filtered_img)

            # if the panels are None, initialize them
            if panelA is None or panelB is None:
                # the first panel will store our original image
                panelA = Label(image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
                # while the second panel will store the edge map
                panelB = Label(image=filtered_img)
                panelB.image = filtered_img
                panelB.pack(side="right", padx=10, pady=10)

            # otherwise, update the image panels
            else:
                # update the panels
                panelA.configure(image=image)
                panelB.configure(image=filtered_img)
                panelA.image = image
                panelB.image = filtered_img

    def update_image(self):
        # grab a reference to the image panels
        global panelA, panelB
        # open a file chooser dialog and allow the user to select an input
        # image

        print(self.brightness_btn_isClicked)
        print(self.contrast_btn_isClicked)
        print(self.blur_isClicked)
        print(self.sobel_x_isClicked)
        print(self.sobel_y_isClicked)
        print(self.sharpen_isClicked)


        output_path = self.path
        self.imgname = output_path.split("/")[-1]
        self.im = Image2(filename=self.imgname)
        self.filter.write_image('output.png')
        output_path = 'output/output.png'

        # ensure a file path was selected
        if len(output_path) > 0:
            # load the image from disk
            filtered_img = cv2.imread(output_path)

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
            filtered_img = cv2.resize(filtered_img, (550, 550))
            # convert the images to PIL format...
            filtered_img = Image.fromarray(filtered_img)
            # ...and then to ImageTk format
            filtered_img = ImageTk.PhotoImage(filtered_img)

            # if the panels are None, initialize them
            if panelB is None:
                # while the second panel will store the edge map
                panelB = Label(image=filtered_img)
                panelB.image = filtered_img
                panelB.pack(side="right", padx=10, pady=10)

            # otherwise, update the image panels
            else:
                # update the panel
                panelB.configure(image=filtered_img)
                panelB.image = filtered_img

    def slider(self):
        if self.brightness_btn_isClicked is True:
            self.filter = transform.brighten(self.im, slider.get())
            self.filter.write_image('testing.png')
            print("Brightness")
            GUI.update_image()

        elif self.contrast_btn_isClicked is True:
            self.filter = transform.adjust_contrast(self.im, slider.get(), 0.5)
            self.filter.write_image('testing.png')
            print("Contrast")
            GUI.update_image()

        elif self.blur_isClicked is True:
            self.filter = transform.blur(self.im, round(slider.get()))
            self.filter.write_image('testing.png')
            print("Blur")
            GUI.update_image()

        elif self.sobel_x_isClicked is True:
            self.filter = transform.apply_kernel(self.im, np.array([[1, 2, 1],
                                                               [0, 0, 0],
                                                               [-1, -2, -1]]))

            self.filter.write_image('testing.png')
            print("Sobel X")
            GUI.update_image()

        elif self.sobel_y_isClicked is True:
            self.filter = transform.apply_kernel(self.im, np.array([[1, 0, -1],
                                                               [2, 0, -2],
                                                               [1, 0, -1]]))

            self.filter.write_image('testing.png')
            print("Sobel Y")
            GUI.update_image()

        elif self.sharpen_isClicked is True:
            self.filter = transform.apply_kernel(self.im, np.array([[0, -1, 0],
                                                               [-1, 5, -1],
                                                               [0, -1, 0]]))

            self.filter.write_image('testing.png')
            print("Sharpen")
            GUI.update_image()

    def adjust_brightness(self):
        self.brightness_btn_isClicked = True
        self.contrast_btn_isClicked = False
        self.blur_isClicked = False
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = False
        self.sharpen_isClicked = False

    def adjust_contrast(self):
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = True
        self.blur_isClicked = False
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = False
        self.sharpen_isClicked = False

    def blur(self):
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = False
        self.blur_isClicked = True
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = False
        self.sharpen_isClicked = False

    def sobel_x(self):
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = False
        self.blur_isClicked = False
        self.sobel_x_isClicked = True
        self.sobel_y_isClicked = False
        self.sharpen_isClicked = False

    def sobel_y(self):
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = False
        self.blur_isClicked = False
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = True
        self.sharpen_isClicked = False

    def sharpen(self):
        self.brightness_btn_isClicked = False
        self.contrast_btn_isClicked = False
        self.blur_isClicked = False
        self.sobel_x_isClicked = False
        self.sobel_y_isClicked = False
        self.sharpen_isClicked = True


if __name__ == '__main__':

    GUI = UserGUI()
    # initialize the window toolkit along with the two image panels
    root = Tk()
    root.title("Image Processing")
    panelA = None
    panelB = None
    # create a button, then when pressed, will trigger a file chooser
    # dialog and allow the user to select an input image; then add the
    # button the GUI
    btn = Button(root, text="Select an image", command=GUI.select_image)
    btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    # slider the GUI
    slider = Scale(root, from_=0, to=20, orient=HORIZONTAL, resolution=0.1)
    slider.pack()

    myframe = LabelFrame(root, text="Image Enhance")
    myframe.pack(side="left", fill="both", expand="yes")

    btn2 = Button(myframe, text="Update", command=GUI.slider).pack()
    btn3 = Button(myframe, text="Adjust Brightness", command=GUI.adjust_brightness).pack()
    btn4 = Button(myframe, text="Adjust Contrast", command=GUI.adjust_contrast).pack()
    btn5 = Button(myframe, text="Blur", command=GUI.blur).pack()
    btn6 = Button(myframe, text="Sobel X", command=GUI.sobel_x).pack()
    btn7 = Button(myframe, text="Sobel Y", command=GUI.sobel_y).pack()
    btn7 = Button(myframe, text="Sharpen", command=GUI.sharpen).pack()


    # kick off the GUI
    root.mainloop()
