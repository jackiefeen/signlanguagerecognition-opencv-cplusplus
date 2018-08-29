# signlanguagerecognition-opencv-cplusplus
Author: Jacqueline Neef

The goal of this project is to combine different Machine Learning and Computer Vision approaches to
create an algorithm that can recognize in real-time the letters A and C of the sign language alphabet.

The DetectSignLanguage code must be compiled run in a Linux environment (x86_64),
where the following opencv libraries are installed:
- libopencv_objdetect
- libopencv_core
- libopencv_highgui
- libopencv_imgcodecs
- libopencv_videoio
- libopencv_imgproc
- libopencv_video
- libopencv_m

Please make sure to use your left hand. The algorithm will recognize it and use the pre-trained Multi-Layer Perceptron to predict if
you are currently showing the letter “A” or “C” of the sign language alphabet.

The given DetectSignLanguage application can also be used to create training data. While the
application is running and the hand is being detected, it is possible to retrieve the
subimage/submatrix of the hand from the Backprojection Matrix. The retrieved image of the hand
will be stored as a 16x16 image in the folder of execution and the pixel values will be written into a
textfile, called letter.txt. By means of chosing the key to press, the retrieved image can be labelled;
i.e. when the key ‘A’ (or ‘C’) is pressed to retrieve the image, the pixel values are labelled as ‘A’ (or ‘C’
respectively).
