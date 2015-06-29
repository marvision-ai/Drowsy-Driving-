#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
//string window_name2 = "window";
int F1, P1;
int F2, P2;

int profileTime1 = 30;
int profileTime2 = 10;


float inital_driverProfile;
double updated_driverProfile;

/** @function main */
int main(int argc, const char** argv)
{
	cv::VideoCapture camera;
	camera.open(0);
	Mat frame;

	//-- 1a. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eye\n"); return -1; };

	//-- 2. While loop for cont. reading of images from queue
	while (true){
		camera >> frame;
		Mat face;
		if (!frame.empty()){
			//-- 3. actual algorithm functions
			detectAndDisplay(frame);
			//imshow("window1", frame);
		}
		else{
			printf(" --(!) No captured frame -- Break!"); break;
		}
		//-- 4. Stop the reading when press c
		int c = waitKey(10);
		if ((char)c == 'c') {
			break;
		}

	}

	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	//________________________________________________________
	//TRACK FRAMES

	if (F1 < profileTime1){
		F1++; //Keep track of how many frames there are 
		printf("INTIAL: %d \n", F1);
	}
	else if (F2 < profileTime2){
		F2++; //Keep track of how many frames there are 
		printf("ACTUAL: %d \n", F2);
	}
	/*else {
	printf("ELSE 1 %d %d %d %d \n", F1,F2,P1,P2);
	F2 = 0;
	}*/
	//________________________________________________________

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// Draw out the circles in the eyes --------------------------------------------------
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);


			//increment the counter 
			if (F1 < profileTime1){
				P1++; //Training Set counter 
				printf("INITAL TRAINING: %d \n", P1);
				break;
			}
			else if (F2 < profileTime2){
				P2++; //Updated set counter 
				printf("ACTUAL TRAINING: %d \n", P2);
				break;
			}
			else {
				printf("ELSE 2 %d %d %d %d \n", F1, F2, P1, P2); \
					//printf("initial driver profile %f", (P1/F1));

					inital_driverProfile = double(P1) / F1; //gives us a percentage 
				updated_driverProfile = double(P2) / F2;

				printf("inital_driverProfile %f \n", inital_driverProfile);
				printf("updated_driverProfile %f \n", updated_driverProfile);

				F2 = 0;
				P2 = 0;
				break;
			}
		}
	}

	//-- Show what you got
	imshow(window_name, frame);
}


