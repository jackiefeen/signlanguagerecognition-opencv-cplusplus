#include <opencv2/core/utility.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/ml.hpp"
#include <iostream>
#include <ctype.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

//author: Jacqueline Neef

//help function
static void help() {
	cout
			<< "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
					"This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
					"It's most known use is for faces.\n"
					"Usage:\n"
					"./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
					"   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
					"   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
					"   [--try-flip]\n"
					"   [filename|camera_index]\n\n"
					"see facedetect.cmd for one call:\n"
					"./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
					"During execution:\n\tHit any key to quit.\n"
					"\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

// function declarations for DetectAndDraw and Camshift
void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale,
		vector<Rect>& faces);

void camshift(Rect& face, Mat& frame1, Ptr<ANN_MLP>& model);

//hard coded cascadeName in order no to have to pass it via the command line
string cascadeName =
		"/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";

// function to load the pre-trained classifier
template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load) {
	// load pre-trained classifier from the specified file
	Ptr<T> model = StatModel::load < T > (filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

int main(int argc, const char** argv) {

	//initialization of objects
	VideoCapture capture;
	Mat frame, image;
	string inputName;
	CascadeClassifier cascade; //cascading boosting classifier object
	double scale;
	vector < Rect > faces;

	//loading the pre-trained classifier for predicting the letter
	Ptr < ANN_MLP > model;
	model = load_classifier<ANN_MLP>("trainedmodel_mlp.xml");

	//command line parser to take in the parameters passed on the command line - if it finds a mistake, help() is called
	cv::CommandLineParser parser(argc, argv,
			"{help h||}"
					"{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
					"{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
					"{scale|1|}{try-flip||}{@filename||}");
	if (parser.has("help")) {
		help();
		return 0;
	}

	scale = parser.get<double>("scale");
	if (scale < 1)
		scale = 1;

	inputName = parser.get < string > ("@filename");
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	if (!cascade.load(cascadeName)) {
		cerr << "ERROR: Could not load classifier cascade" << endl;
		help();
		return -1;
	}
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1)) {
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
			cout << "Capture from camera #" << camera << " didn't work" << endl;
	}

	else {
		cout << "There seems to be an error with starting the facedetect"
				<< endl;
	}

	if (capture.isOpened()) {
		cout << "Video capturing has been started ..." << endl;

		// in an infinite loop, retrieve the current frame from the camera and call DetectAndDraw/Camshift
		for (;;) {
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();

			// if no face has been detected yet - call detectAndDrae on the current frame
			if (faces.empty()) {
				//cout << "faces seems to be empty right now, let's find some "<< endl;
				detectAndDraw(frame1, cascade, scale, faces);

				// if a face has been detected - track it in the current frame using camshift
			} else {
				//cout << "Here is a face! Let's keep it in the focus " << endl;
				//call camshift with the first element in the faces vector
				camshift(faces[0], frame1, model);
			}

			//set the key to break the infinite loop and terminate the sign language detection
			char c = (char) waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}

	} else {
		cout << "Camera doesnt seem to work " << endl;
	}

	return 0;
}

//definition of the detectAndDraw function to recognize a face in the picture
void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale,
		vector<Rect>& faces) {
	double t = 0;

	Mat gray, smallImg;
	const static Scalar colors[] = { Scalar(255, 0, 0), Scalar(255, 128, 0),
			Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 128, 255), Scalar(
					0, 255, 255), Scalar(0, 0, 255), Scalar(255, 0, 255) };

	//convert colors to greyscale
	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	//make the size of the image smaller by using linear interpolation (-> for blurring)
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double) getTickCount();
	//detectMultiScale: launch the AdaBoost algorithm
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE,
			Size(30, 30));

	t = (double) getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency());

	//going through the faces array in order to extract the rectangle
	for (size_t i = 0; i < faces.size(); i++) {
		Rect r = faces[i];
		Point center;
		Scalar color = colors[i % 8]; // only 8 colors are used
		int radius;

		// the aspect_ratio makes sure that the face is within a biologically common shape of a face
		double aspect_ratio = (double) r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
			center.x = cvRound((r.x + r.width * 0.5) * scale);
			center.y = cvRound((r.y + r.height * 0.5) * scale);
			radius = cvRound((r.width + r.height) * 0.25 * scale);
			circle(img, center, radius, color, 3, 8, 0);
		} else
			rectangle(img, cvPoint(cvRound(r.x * scale), cvRound(r.y * scale)),
					cvPoint(cvRound((r.x + r.width - 1) * scale),
							cvRound((r.y + r.height - 1) * scale)), color, 3, 8,
					0);
	}
	//imshow("result", img);
}

// definition of the camshift function to keep focused on the faces
void camshift(Rect& selection, Mat& frame1, Ptr<ANN_MLP>& model) {

	//initialization of variables and objects
	//flow of code
	int trackObject = -1;
	bool selectObject = false;
	bool backprojMode = false;
	bool showHist = false;

	//camshift variables/objects
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	Rect trackWindow;

	int vmin = 10, vmax = 256, smin = 30;

	//UI: create the windows that are shown during the execution
	namedWindow("Histogram", 0);
	namedWindow("Backprojection", 0);
	namedWindow("Tracking", 0);
	namedWindow("HandWindow", 0);

	//instantiation of matrix objects
	Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

	bool paused = false;

	if (!paused) {
		//convert the current frame to the hsv color space
		cvtColor(frame1, hsv, COLOR_BGR2HSV);

		if (trackObject) {
			int _vmin = vmin, _vmax = vmax;

			inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
					Scalar(180, 256, MAX(_vmin, _vmax)), mask);
			int ch[] = { 0, 0 };
			hue.create(hsv.size(), hsv.depth());
			mixChannels(&hsv, 1, &hue, 1, ch, 1);

			//if camshift is called for the first time on the current selection
			if (trackObject < 0) {
				// Set up the CAMShift search properties based on the color histogram model
				// in order to recognize a face based on the color scheme
				Mat roi(hue, selection), maskroi(mask, selection);
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
				normalize(hist, hist, 0, 255, NORM_MINMAX);

				//set the trackWindow to the selection obtained by DetectAndDraw
				trackWindow = selection;
				trackObject = 1; // Don't set up again, unless user selects new ROI

				histimg = Scalar::all(0);
				int binW = histimg.cols / hsize;
				Mat buf(1, hsize, CV_8UC3);
				for (int i = 0; i < hsize; i++)
					buf.at < Vec3b > (i) = Vec3b(
							saturate_cast < uchar > (i * 180. / hsize), 255,
							255);
				cvtColor(buf, buf, COLOR_HSV2BGR);

				for (int i = 0; i < hsize; i++) {
					int val = saturate_cast<int>(
							hist.at<float>(i) * histimg.rows / 255);
					rectangle(histimg, Point(i * binW, histimg.rows),
							Point((i + 1) * binW, histimg.rows - val),
							Scalar(buf.at < Vec3b > (i)), -1, 8);
				}
			}

			// If Camshift is called subsequent times on the current selection

			//calculate the backprojection, which is the probability that the sought face histogram
			//matches with the given pixel
			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
			backproj &= mask;
			//apply camshift to the backprojection matrix in order to find the face in the image
			RotatedRect trackBox = CamShift(backproj, trackWindow,
					TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10,
							1));

			//retrieve the bounding rectangle of the current trackBox around the face
			Rect box = trackBox.boundingRect();
			//rectangle(frame1, box, Scalar(255, 0, 255), 2, 1);
			int rightborder = box.x + box.width; //retreive the right border of the detected face
			int bottom = box.y + box.height;//retrieve the lower border of the detected face

			//write all points in the backprojection matrix that are on the face to probability = 0
			//this will lead to detecting the hand instead of the face
			for (int xaxis = box.x; xaxis < rightborder; xaxis++) {
				for (int yaxis = box.y; yaxis < bottom; yaxis++) {
					backproj.at < uchar > (yaxis, xaxis) = 0;
				}
			}

			//set the trackWindow coordinates to the corridor on the left of the detected face
			//I set it on the left, because I would like to use my left hand for the sign language
			trackWindow.x = rightborder + 20;
			trackWindow.y = 0;

			// set the width of the trackWindow equal to the width of the detected face
			//and to the height of the entire frame
			trackWindow.width = rightborder;
			trackWindow.height = frame1.rows;

			//draw a blue rectangle around the set trackWindow
			rectangle(frame1, trackWindow, Scalar(255, 0, 0), 2, 1);

			//perform CamShift again with the new parameters for tracking the hand
			RotatedRect handBox;
			handBox = CamShift(backproj, trackWindow,
					TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10,
							1));

			// rotate the handBox and make sure that it does now flow out of the frame
			Rect box2 = handBox.boundingRect();
			box2 &= Rect(0, 0, frame1.cols, frame1.rows);

			//draw a red ellipse around the detected hand
			ellipse(frame1, handBox, Scalar(0, 0, 255), 2, 1);

			//extract a submatrix from the image where the hand is detected
			Mat hand = backproj(box2);
			//resize the submatrix to 16x16 pixels by using linear interpolation
			resize(hand, hand, Size(16, 16), 0, 0, INTER_LINEAR);

			//reshape the hand matrix to one row
			//and convert the pixels to floats - the pixel can have any value between 0-1.0
			Mat img = hand.reshape(0, 1);
			img.convertTo(img, CV_32F);

			//predict the ASCII code of the showed letter in the image using the pre-trained model
			float r = model->predict(img);
			r = r + (int) ('A');

			//print the prediction to the console
			if (r == 65) {
				cout << "You are showing: 'A'" << endl;
			}
			if (r == 67) {
				cout << "You are showing: 'C'" << endl;
			}

			// If the keys A or C are pressed --> Generate training data
			char c = (char)waitKey(10);
			if(c == 'a' || c == 'c'){

			//create a random filename for the extracted hand images (to not overwrite the image before when saving)
			int label = 0;
			label = label + rand() % 1000;
			string save = "handImage" + std::to_string(label)
					+ ".jpg";

			//create a fileoutputstream to write the pixels of the extracted hand image into the textfile letter.txt
			ofstream os("letter.txt", ios::out | ios::app);

			//depending on the letter pressed, the saved pixels are labelled with "A" or "C"
			switch (c) {
			case 'a':
				//show the retrieved hand image in a window and save it to a file
				imshow("HandWindow", hand);
				imwrite(save, hand);
				//write the pixels into letter.txt
				os << "A,";
				os << format(img, Formatter::FMT_CSV) << endl;
				os.close();
			case 'c':
				//show the retrieved hand image in a window and save it to a file
				imshow("HandWindow", hand);
				imwrite(save, hand);
				//write the pixels into letter.txt
				os << "C,";
				os << format(img, Formatter::FMT_CSV) << endl;
				os.close();
			}
			}

			if (trackWindow.area() <= 1) {
				int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols,
						rows) + 5) / 6;
				trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
						trackWindow.x + r, trackWindow.y + r)
						& Rect(0, 0, cols, rows);
			}

			if (backprojMode)
				cvtColor(backproj, frame1, COLOR_GRAY2BGR);

		}
	} else if (trackObject < 0)
		paused = false;

	if (selectObject && selection.width > 0 && selection.height > 0) {
		Mat roi(frame1, selection);
		bitwise_not(roi, roi);
	}

	//display the defined windows
	imshow("Tracking", frame1);
	imshow("Histogram", histimg);
	imshow("Backprojection", backproj);
}
