#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>


/*------------Constant------------*/
#define MORPH_MASK      10                  // 모폴로지 연산 mask 사이즈 n X n
#define GAUSSIAN_MASK   7                   // 가우시안 연산 mask 사이즈 n X n
#define MA_FILTER       10                  // moving average filter 사이즈

// Chessboard Parameter
#define CHESSBOARDGRID  0.025               // Chessboard 사각형 한 변 길이 [m]
#define CHESS_ROWS      8                   // Chessboard 헁 꼭지점 수 (행 사각형 갯수-1)
#define CHESS_COLS      6                   // Chessboard 열 꼭지점 수 (열 사각형 갯수 1)

// Camera Intrinsic Parameter
#define FX              786.5830110000001   // focal length_x
#define FY              797.147364          // focal length_y
#define CX              390.533843          // principal point_x
#define CY              322.382708          // principal point_y
#define K1              0.06600499999999999 // radial distortion
#define K2              -0.064593           // radial distortion
#define P1              0.006004            // tangential distortion
#define P2              -0.000721           // tangential distortion

#define I2W             1                   // image-->world
#define W2I             0                   // world-->image

#define LD              287                 // Lookahead Distance 1.5[m]일때 y축 픽셀좌표
#define SECTION1        348                 // SECTION1 시작 y좌표 (1.3[m]) //348     
#define SECTION2        287                 // SECTION2 시작 y좌표 (1.5[m]) //286.5
#define SECTION3        240                 // SECTION3 시작 y좌표 (1.7[m]) //240
#define SECTIONEND      204                 // SECTION3 끄ㅌ y좌표 (1.9[m]) //204
#define LANEWIDTH       0.86                 // 차선 폭[m]

#define HOUGH           0                   // HoughLines
#define HOUGHP          1                   // HoughLinesP

#define FONTSIZE        6                   // 좌표 text 사이즈

// Scalar 색상
const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar GREEN(0, 180, 0);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar WHITE(255, 255, 255);
const cv::Scalar YELLOW(0, 255, 255);
const cv::Scalar CYAN(255, 255, 0);
const cv::Scalar MAGENTA(255, 0, 255);
const cv::Scalar GRAY(150, 150, 150);
const cv::Scalar PURPLE(255, 0, 127);


/*------------Struct------------*/
typedef struct{
	double b;       // a fraction between 0 and 1
	double g;       // a fraction between 0 and 1
	double r;       // a fraction between 0 and 1
}bgr;

typedef struct{
	double h;       // angle in degrees
	double s;       // a fraction between 0 and 1
	double v;       // a fraction between 0 and 1
}hsv;
 
typedef struct{
    double X;               // 카메라 X위치 [m]  
    double Y;               // 카메라 Y위치 [m]  
    double Z;               // 카메라 Z위치 [m]  
    double pan;             // 카메라 좌우 회전각 [rad] (왼쪽 +, 오른쪽 -)
    double tilt;            // 카메라 상하 회전각 [rad] (위  +,  아래  -)
    //double roll;          // 카메라 광학축 기준 회전각 [rad] (카메라와 같은 방향을 바라볼 때, 시계방향 +, 반시계방향 -)
    cv::Mat Rotation;       // 카메라 외부 파라미터 - Rotation Matrix
    cv::Mat Translation;    // 카메라 외부 파라미터 - Translation Matrix
    cv::Mat cameraMatrix;   // 카메라 내부 파라미터 - Focal Length, Principal Point
    cv::Mat distCoeffs;     // 카메라 내부 파라미터 - Distortion Coefficients
}camParam;                  // 카메라 파라미터 구조체


/*------------Global Variable------------*/
camParam camera;
cv::Size patternsize(CHESS_ROWS,CHESS_COLS); //checker board interior number of corners

cv::Mat img_comb;
std::vector<cv::Point> poly, roi1, roi2, roi3;
cv::Point2f old_s1, old_wp;
static int count = 0;

// MouseClick
cv::Mat img_click;
std::vector<cv::Point2f> clicked;
static int click_cnt=1;

std::vector<cv::Point2f> MAF_buf, SEC1_buf, SEC2_buf, SEC3_buf, BOTTOM_buf;   // Moving Average Filter 담는 벡터

// Save Video Parameter
const double fps = 30.0;            // 비디오 프레임수
int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4'); // 비디오 코덱 'M', 'P', '4', 'V'
cv::VideoWriter video_line("line_test54.mp4", fourcc, fps, cv::Size(800, 600), true);   // 비디오 파일명과 사이즈 등
// cv::VideoWriter video_ori("ori_test12.mp4", fourcc, fps, cv::Size(800, 600), true);   // 비디오 파일명과 사이즈 등
// cv::VideoCapture cap("yellow_line_test44.mp4");      
cv::VideoCapture cap("ori_test54.mp4");     

// putText Parameter
static int font = cv::FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
static double fontScale = FONTSIZE / 10.0;
static int thickness = (int)fontScale + 2;
static int baseLine = 0;

// Camera Intrinsic Parameter
double camera_matrix[] = {FX,0.,CX,0.,FY,CY,0.,0.,1.};
cv::Mat cameraMatrix(3, 3, CV_64FC1, camera_matrix);

// Camera Distortion Coefficients
double distortion_coeffs[] = {K1, K2, P1, P2};
cv::Mat distCoeffs(4, 1, CV_64FC1, distortion_coeffs);

static bool cali_flag = 1;  // 첫 실행 확인 flag

// Trackbar Variable(노란색)
static int lowH = 0, highH = 75;
static int lowS = 55, highS = 255;
static int lowV = 71, highV = 255;

// // Trackbar Variable(노란색)
// static int lowH = 0, highH = 75;
// static int lowS = 30, highS = 255;
// static int lowV = 100, highV = 255;

// // Trackbar Variable(검정색)
// static int lowH = 0, highH = 179;
// static int lowS = 0, highS = 255;
// static int lowV = 109, highV = 255;

// Canny edge Threshold
static int lowTH = 50, highTH = 150;

// Hough Threshold
static int houghTH = 40;
static int houghPTH = 25, minLine = 15, maxGap=25;


/*------------Fuction Prototype------------*/
void ImageCallback(const sensor_msgs::Image::ConstPtr &img);            // 이미지 subscribe 될 때 호출되는 함수
void MouseCallback(int event, int x, int y, int flags, void *userdata); // 창 마우스 클릭될 때 호출되는 함수

void setROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& range);   // Rect 함수로 사각형 ROI설정함수
void setROI(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& points);    // fillpoly함수로 다각형 ROI설정 함수
void setROIGray(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& points);    // fillpoly함수로 다각형 ROI설정 함수
bgr hsv2bgr(hsv in);    //hsv->rgb로 바꾸는 함수
hsv bgr2hsv(bgr in);    //rgb->hsv로 바꾸는 함수
void checkHSV(const cv::Mat& img, const cv::Point2f& point);
void Convert_Binary(const cv::Mat& img, cv::Mat& img_binary, bool show_trackbar = false);   // H,S,V 범위에 따라 이진화하는 함수  (input image, output image)  
void Edge_Detect(const cv::Mat& img_gray, cv::Mat& img_edge, bool show_trackbar = false);   // Gray이미지 받아서 edge 찾는 함수  (input image, output image)  
void Find_Line(const cv::Mat& img_edge, const cv::Mat& img_line,  bool show_trackbar = false, bool modeP = HOUGHP);       // 직선 찾는 함수(input image, output image)  
void Find_FinalLine(const cv::Mat& img_edge, const cv::Mat& img_line,  bool show_trackbar = false, bool modeP = HOUGHP);  // 대표 차선 찾는 함수(input image, output image)  
void Final_Line(const cv::Mat& img_edge, std::vector<cv::Point2f>& left, std::vector<cv::Point2f>& right, cv::Scalar Lcolor = RED, cv::Scalar Rcolor = BLUE);
void Line_detect(const cv::Mat& img_edge, const cv::Mat& img_draw, bool show_trackbar = false);
cv::Point2f VanishingPoint(const std::vector<cv::Point2f>& leftLine, const std::vector<cv::Point2f>& rightLine);                // 소실점 찾는 함수
cv::Point2f MovingAverageFilter(const cv::Point& array, std::vector<cv::Point2f>& buf = MAF_buf, size_t filter_size = MA_FILTER);    // Moving Average Filter point에 적용하는 함수                    
void Find_Contours(const cv::Mat& img_edge, cv::Mat& img_contour);                          // 테두리 찾는 함수                 (input image, output image)  
void Histogram(const cv::Mat& img_binary);

void makeobjectPoints(std::vector<cv::Point3f>& obj, const int mode, const bool show_detail = false);   // Chessboard 실제 좌표(World Coordinate) 입력 함수
void ChessBoard(const cv::Mat& img,  std::vector<cv::Point2f>& corners , bool show_board = false);      // Chessboard 꼭지점 찾는 함수 (input image, output vector)
void drawXYZ(const cv::Mat& img, const camParam& camera, const bool mode,                               // Chessboard 기준 좌표계에 X,Y,Z 축 그리는 함수 (input image, input camParam)
             double x = 0.15, double y = 0.2, double z = 0.025);                                        
void ExtrinsicCalibration(const cv::Mat& img, camParam& camera, bool show_cali = false);                // 카메라 캘리브레이션 함수 (input image, output camParam)

cv::Point2f transformPoint(const cv::Point2f& cur, const cv::Mat& T);   // 좌표계 변환 함수
void Projection(const cv::Point2f& src, cv::Point2f& dst,
                bool direction = I2W);  // 픽셀좌표계-->월드좌표계 , 월드좌표계-->픽셀좌표계 변환하는 함수 (input vector, output vector, 변환 방법)
void Projection(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst,
                bool direction = I2W);  // 픽셀좌표계-->월드좌표계 , 월드좌표계-->픽셀좌표계 변환하는 함수 (input vector, output vector, 변환 방법)
void BirdEyeView(const cv::Mat& src, cv::Mat& dst);
void BirdEyeView(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point2f>& imagePoints);    // 4개의 점입력하면 해당 점을 펼쳐주는 함수


/*------------Fuction Expression------------*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub_img = it.subscribe("/image_raw", 1, ImageCallback);

    if(!cap.isOpened()){
        std::cout << "Can't open the video!";
    }

    ROS_INFO("Hello World!");
    
    ros::spin();
}

void ImageCallback(const sensor_msgs::Image::ConstPtr &img){
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img_bgr, img_binary, img_edge, img_line, img_contour, img_save, img_warp, img_read, img_roi;
    std::vector<cv::Point2f> srcPts, dstPts;
    cv::Point2f lRange;
    //ROS_INFO("IMAGE(%d, %d)", img->width, img->height);
    
    try {
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        img_bgr = cv_ptr->image;
        
        cap >> img_bgr;
        // resize(img_bgr,img_bgr, cv::Size(800,600));
        // cv::waitKey(100);
        
        // img_bgr = cv::imread("yellow_test.jpg");
        // img_read = cv::imread("photo_0.jpg");
        // if(img_read.empty()){
        //     std::cout << "img empty!" << std::endl;
        // }
        // //ROS_INFO("IMAGE(%d, %d)", img_read.cols, img_read.rows);
        // resize(img_read,img_read, cv::Size(1008,756));
        // img_bgr = img_read;
        // img_click = img_bgr.clone(); //클릭이미지 초기화
        
        if(cali_flag == 1){
            // Projection(cv::Point2f{0,1.30},lRange,W2I); //348
            // Projection(cv::Point2f{0,1.50},lRange,W2I); //286.5
            // Projection(cv::Point2f{0,1.70},lRange,W2I); //240
            // Projection(cv::Point2f{0,1.90},lRange,W2I); //204
            // Projection(cv::Point2f{0,2.50},lRange,W2I); //132
            // Projection(cv::Point2f{0,3.00},lRange,W2I); //94

            // img_click = img_bgr.clone();

            poly.clear();
            poly.push_back(cv::Point(0, SECTION1)); // 161
            poly.push_back(cv::Point(img_bgr.cols,SECTION1)); // 161
            poly.push_back(cv::Point(img_bgr.cols,SECTIONEND)); // 221
            poly.push_back(cv::Point(0,SECTIONEND));   //221

            roi1.clear();   // 화면 제일 하단 부분(130~150cm)
            roi1.push_back(cv::Point(0, SECTION1)); // 161
            roi1.push_back(cv::Point(img_bgr.cols,SECTION1)); // 161
            roi1.push_back(cv::Point(img_bgr.cols,SECTION2)); // 221
            roi1.push_back(cv::Point(0,SECTION2));   //221

            roi2.clear();   // 화면 중간 부분(150~190cm)
            roi2.push_back(cv::Point(0, SECTION2)); // 161
            roi2.push_back(cv::Point(img_bgr.cols,SECTION2)); // 161
            roi2.push_back(cv::Point(img_bgr.cols,SECTION3)); // 221
            roi2.push_back(cv::Point(0,SECTION3));   //221

            roi3.clear();   // 화면 제일 상단 부분(190~250cm)
            roi3.push_back(cv::Point(0, SECTION3)); // 161
            roi3.push_back(cv::Point(img_bgr.cols,SECTION3)); // 161
            roi3.push_back(cv::Point(img_bgr.cols,SECTIONEND)); // 221
            roi3.push_back(cv::Point(0,SECTIONEND));   //221

            // setROI(img_bgr,img_roi,poly);

            //ExtrinsicCalibration(img_bgr, camera, cali_flag);
        }
        // setROI(img_bgr,img_roi,cv::Rect(cv::Point(0, 161), 
        //                 cv::Point(img_bgr.cols, 221)));
        setROI(img_bgr,img_roi,poly);
        Convert_Binary(img_roi, img_binary, true);
        // cv::cvtColor(img_roi,img_binary,CV_BGR2GRAY);
        Edge_Detect(img_binary, img_edge, true);
        // Find_FinalLine(img_edge,img_bgr, true, HOUGHP);
        Line_detect(img_edge, img_bgr, true);

        // resize(img_bgr,img_save, cv::Size(800,600));
        // cv::cvtColor(img_save,img_save,CV_GRAY2BGR);
        // video_ori << img_bgr;   // 저장할 영상 이미지

        // cv::imwrite("yellow_test.jpg", img_bgr);
        // std::cout << " save img "  << std::endl;

        // ExtrinsicCalibration(img_bgr, camera);   
        // BirdEyeView(img_bgr, img_warp);
        // cv::line(img_bgr, cv::Point{0,187}, cv::Point{img_bgr.cols,187}, GREEN, 5);
        // cv::line(img_bgr, cv::Point{0,161}, cv::Point{img_bgr.cols,161}, RED, 2);
        // cv::line(img_bgr, cv::Point{0,221}, cv::Point{img_bgr.cols,221}, BLUE, 2);
        count++;
        cali_flag = 0;
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Error to convert!");
        return;
    }
    // cv::imshow("Test_click", img_bgr); // 이미지에서 클릭한 점 좌표 알아내는것
    // cv::setMouseCallback("Test_click", MouseCallback);
    // cv::imshow("Image Show", img_bgr);
    // cv::imshow("Image ROI", img_roi);
    // cv::imshow("Image warp", img_warp);
    // cv::imshow("Image edge", img_edge);
    cv::waitKey(1);
;}

void MouseCallback(int event, int x, int y, int flags, void *userdata){
    if(event==CV_EVENT_LBUTTONDOWN){
        cv::Mat img_bev;
        cv::Point2f clickP = {(float)x,(float)y};
        cv::Point2f dstP;
        
        // checkHSV(img_click, clickP);
        
        std::cout << "clicked >> x = " << x << ", y = " << y << std::endl;
        Projection(clickP, dstP);
        // if(click_cnt%4 == 0){
        //     BirdEyeView(img_click, img_bev, clicked);
        //     clicked.clear();
        // }

        std::string coord = "(" + std::to_string((int)(dstP.x*100)) + "," + std::to_string((int)(dstP.y*100)) + ") cm";
	    cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	    cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	    word_center.x = clickP.x - (size.width / 2);
	    word_center.y = clickP.y + (size.height);
        cv::circle(img_click, clickP, 2, BLUE, -1);
	    cv::putText(img_click, coord, word_center, font, fontScale, BLACK, thickness, 8);
        
        cv::imshow("check clicked", img_click);
        cv::waitKey(1);
        click_cnt++;
    }
}

void setROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& range){
    cv::Point tl = range.tl();
    cv::Point br = range.br();

    dst = src(cv::Rect(tl,br));         
}

void setROI(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& points){
    std::vector<std::vector<cv::Point>> poly(1, std::vector<cv::Point>());
    cv::Mat mask = cv::Mat::zeros(cv::Size(src.cols, src.rows),CV_8UC3);

    for(int i=0; i<points.size(); i++){
        poly[0].push_back(points[i]);
    }
    const cv::Point *pt = (const cv::Point *) cv::Mat(poly[0]).data;
    const cv::Point *polygon[1] = {pt};
    int npts[1] = {(int)points.size()};

    cv::fillPoly(mask, polygon, npts, 1, WHITE);
    // cv::polylines(mask, polygon, npts, 1, true, WHITE);

    cv::bitwise_and(src, mask, dst);
    // cv::imshow("fillpoly", dst);
    // cv::waitKey(1);
     
}

void setROIGray(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point>& points){
    std::vector<std::vector<cv::Point>> poly(1, std::vector<cv::Point>());
    cv::Mat mask = cv::Mat::zeros(cv::Size(src.cols, src.rows),CV_8UC1);

    for(int i=0; i<points.size(); i++){
        poly[0].push_back(points[i]);
    }
    const cv::Point *pt = (const cv::Point *) cv::Mat(poly[0]).data;
    const cv::Point *polygon[1] = {pt};
    int npts[1] = {(int)points.size()};

    cv::fillPoly(mask, polygon, npts, 1, WHITE);
    // cv::polylines(mask, polygon, npts, 1, true, WHITE);

    cv::bitwise_and(src, mask, dst);
    // cv::imshow("fillpoly", dst);
    // cv::waitKey(1);
     
}

bgr hsv2bgr(hsv in)
{
	double      hh, p, q, t, ff;
	long        i;
	bgr         out;

	if (in.s <= 0.0) {       // < is bogus, just shuts up warnings
		out.r = in.v;
		out.g = in.v;
		out.b = in.v;
		return out;
	}
	hh = in.h;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = (long)hh;
	ff = hh - i;
	p = in.v * (1.0 - in.s);
	q = in.v * (1.0 - (in.s * ff));
	t = in.v * (1.0 - (in.s * (1.0 - ff)));

	switch (i) {
	case 0:
		out.r = in.v;
		out.g = t;
		out.b = p;
		break;
	case 1:
		out.r = q;
		out.g = in.v;
		out.b = p;
		break;
	case 2:
		out.r = p;
		out.g = in.v;
		out.b = t;
		break;

	case 3:
		out.r = p;
		out.g = q;
		out.b = in.v;
		break;
	case 4:
		out.r = t;
		out.g = p;
		out.b = in.v;
		break;
	case 5:
	default:
		out.r = in.v;
		out.g = p;
		out.b = q;
		break;
	}
	return out;
}

hsv bgr2hsv(bgr in)
{
	hsv        out;
	double      min, max, delta;

	min = in.r < in.g ? in.r : in.g;
	min = min  < in.b ? min : in.b;

	max = in.r > in.g ? in.r : in.g;
	max = max  > in.b ? max : in.b;

	out.v = max;                                // v
	delta = max - min;
	if (max > 0.0) { // NOTE: if Max is == 0, this divide would cause a crash
		out.s = (delta / max);                  // s
	}
	else {
		// if max is 0, then r = g = b = 0
		// s = 0, v is undefined
		out.s = 0.0;
		out.h = 0.0;                            // its now undefined
		return out;
	}
	if (in.r >= max)                           // > is bogus, just keeps compilor happy
		if (delta == 0) {
			out.h = 0.0;
		}
		else {
			out.h = (in.g - in.b) / delta;        // between yellow & magenta
		}
	else
		if (in.g >= max)
			out.h = 2.0 + (in.b - in.r) / delta;  // between cyan & yellow
		else
			out.h = 4.0 + (in.r - in.g) / delta;  // between magenta & cyan

	out.h *= 60.0;                              // degrees

	if (out.h < 0.0)
		out.h += 360.0;

	return out;
}

void checkHSV(const cv::Mat& img, const cv::Point2f& point){
    bgr color;
    int b = img.at<cv::Vec3b>(point.y, point.x)[0];
    int g = img.at<cv::Vec3b>(point.y, point.x)[1];
    int r = img.at<cv::Vec3b>(point.y, point.x)[2];

    std::cout << " b : " << b;
    std::cout << " g : " << g;
    std::cout << " r : " << r << std::endl;

    color.b = b/255.0;
    color.g = g/255.0;
    color.r = r/255.0;

    hsv whathsv = bgr2hsv(color);

    std::cout << " H : " << whathsv.h;
    std::cout << " S : " << whathsv.s*255;
    std::cout << " v : " << whathsv.v*255 << std::endl;
}

void Convert_Binary(const cv::Mat& img, cv::Mat& img_binary, bool show_trackbar){
    if(show_trackbar){
        cv::createTrackbar("Low_H", "Binary",  &lowH, 179);
        cv::createTrackbar("High_H","Binary", &highH, 179);
        cv::createTrackbar("Low_S", "Binary",  &lowS, 255);
        cv::createTrackbar("High_S","Binary", &highS, 255);
        cv::createTrackbar("Low_V", "Binary",  &lowV, 255);
        cv::createTrackbar("High_V","Binary", &highV, 255);
    }
    
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(img_hsv, cv::Scalar(lowH, lowS, lowV), cv::Scalar(highH, highS, highV), img_binary);

    // Opening(침식->팽창), 작은점들 제거
    cv::erode(img_binary, img_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_MASK,MORPH_MASK))); 
    cv::dilate(img_binary, img_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_MASK,MORPH_MASK)));

    // Closing(팽창->침식), 구멍 메우기
    cv::erode(img_binary, img_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_MASK,MORPH_MASK)));
    cv::dilate(img_binary, img_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(MORPH_MASK,MORPH_MASK)));

    cv::imshow("Binary", img_binary);
    cv::waitKey(1);
}

void Edge_Detect(const cv::Mat& img_gray, cv::Mat& img_edge, bool show_trackbar){
    if(show_trackbar){
        cv::createTrackbar("Low_Threshold", "Edge",  &lowTH, 300);
        cv::createTrackbar("High_Threshold","Edge", &highTH, 300);
    }

    img_edge = img_gray.clone();
    cv::GaussianBlur(img_edge, img_edge, cv::Size(GAUSSIAN_MASK,GAUSSIAN_MASK), 0, 0);
    cv::Canny(img_edge, img_edge, lowTH, highTH);
    
    cv::imshow("Edge", img_edge);
    cv::waitKey(1);
}

void Find_Line(const cv::Mat& img_edge, const cv::Mat& img_line, bool show_trackbar, bool modeP){
    cv::Mat img_draw = img_line.clone();
    cv::Point pt1, pt2;

    if(modeP){
        if(show_trackbar){
            cv::createTrackbar("HoughLinesP_Threshold", "Find Line", &houghPTH, 179);
            cv::createTrackbar("minLineLength",         "Find Line",  &minLine, 179);
            cv::createTrackbar("maxLineGap",            "Find Line",   &maxGap, 179);
        }
        std::vector<cv::Vec4i> lines;

        cv::HoughLinesP(img_edge, lines, 1, CV_PI/180., houghPTH, minLine, maxGap);
        // std::cout << " line size : " << lines.size() << std::endl;

        for(int i=0; i<lines.size(); i++){
            pt1 = {lines[i][0],lines[i][1]};
            pt2 = {lines[i][2],lines[i][3]};

            // std::cout << " pt1 (" << pt1.x << ", " << pt1.y << ")" << std::endl;
            // std::cout << " pt2 (" << pt2.x << ", " << pt2.y << ")" << std::endl;
        
            cv::line(img_draw, pt1, pt2, RED, 2, 8);
        }
    }
    else{
        if(show_trackbar){
            cv::createTrackbar("HoughLines_Threshold", "Find Line",  &houghTH, 179);
        }
        std::vector<cv::Vec2f> lines;

        cv::HoughLines(img_edge, lines, 1, CV_PI/180., houghTH);
    
        for(int i=0; i<lines.size(); i++){
            float rho = lines[i][0], theta = lines[i][1];
            
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;

            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));

            cv::line(img_draw, pt1, pt2, RED, 2, 8);
        }
    }
    cv::imshow("Find Line", img_draw);
    cv::waitKey(1);
}

void Find_FinalLine(const cv::Mat& img_edge, const cv::Mat& img_line, bool show_trackbar, bool modeP){
    cv::Mat img_draw = img_line.clone();
    cv::Point2f lp_old, wp_old;
    float cx = img_line.cols/2.;
    float qx = img_line.cols/4.;

    if(modeP){
        if(show_trackbar){
            cv::createTrackbar("HoughLinesP_Threshold", "Find Line", &houghPTH, 179);
            cv::createTrackbar("minLineLength",         "Find Line",  &minLine, 179);
            cv::createTrackbar("maxLineGap",            "Find Line",   &maxGap, 179);
        }
        std::vector<cv::Vec4i> lines;

        cv::HoughLinesP(img_edge, lines, 1, CV_PI/180., houghPTH, minLine, maxGap);

        float slope_threshold = 0.5;
        std::vector<cv::Vec4i> rightLines, leftLines, remainR, remainL; // 좌우 차선 분리하여 담을 벡터(선분의 시작좌표(x1,y1),끝좌표(x2,y2) 4개 데이터 들어있음)
        float finalLine[2][4];  // 최종 차선 담을 배열
        cv::Point pt1, pt2, rightP1, rightP2, leftP1, leftP2;
        std::vector<cv::Point2f> left,right;

        for(int i=0; i<lines.size(); i++){
            int x1 = lines[i][0];
            int y1 = lines[i][1];
            int x2 = lines[i][2];
            int y2 = lines[i][3];

            float slope; // 선분 기울기

            if(x2-x1 == 0)  slope = 999.9;  // 분모가 0이면 기울기 거의 무한대
            else    slope = (y2-y1)/(float)(x2-x1);

            if(fabs(slope) > slope_threshold){ // 너무 수평인 직선들 제거
                if((slope>0) && (x1>cx) && (x2>cx)){ //(x1>cx+qx) && (x2>cx+qx)
                    // 기울기가 양수이고 화면 오른쪽에 위치하면 우측차선으로 분류
                    rightLines.push_back(lines[i]);
                }
                else if((slope<0) && (x1<cx) && (x2<cx)){ //(x1<cx-qx) && (x2<cx-qx)
                    // 기울기가 음수이고 화면 왼쪽에 위치하면 좌측차선으로 분류
                    leftLines.push_back(lines[i]);
                }
                else if((x1>cx) && (x2>cx)){
                    // 기울기는 음수이지만 화면 오른쪽에 위치한 차선
                    remainR.push_back(lines[i]);
                }
                else if((x1<cx) && (x2<cx)){
                    // 기울기는 양수이지만 화면 왼쪽에 위치한 차선
                    remainL.push_back(lines[i]);
                }
            }
            // std::cout << " pt1 (" << pt1.x << ", " << pt1.y << ")" << std::endl;
            // std::cout << " pt2 (" << pt2.x << ", " << pt2.y << ")" << std::endl;
        }

        // 좌우 차선 둘 다 잘 검출된 경우(직진)
        if((leftLines.size()!=0) && (rightLines.size()!=0)){    // 좌우 차선이 검출 되었는지 확인
            // 최종차선 초기갑ㅅ 설정
            finalLine[0][0] = leftLines[0][0];
            finalLine[0][1] = leftLines[0][1];
            finalLine[0][2] = leftLines[0][2];
            finalLine[0][3] = leftLines[0][3];
            finalLine[1][0] = rightLines[0][0];
            finalLine[1][1] = rightLines[0][1];
            finalLine[1][2] = rightLines[0][2];
            finalLine[1][3] = rightLines[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int l=0; l<leftLines.size(); l++){
                if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
                else    slope = (leftLines[l][3]-leftLines[l][1])/
                         (float)(leftLines[l][2]-leftLines[l][0]);

                if(slope < lslope){
                    // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[0][0] = leftLines[l][0];
                    finalLine[0][1] = leftLines[l][1];
                    finalLine[0][2] = leftLines[l][2];
                    finalLine[0][3] = leftLines[l][3];

                    lslope = slope;
                }
            }
    
            for(int r=0; r<rightLines.size(); r++){
                if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
                else    slope = (rightLines[r][3]-rightLines[r][1])/
                         (float)(rightLines[r][2]-rightLines[r][0]);

                if(slope > rslope){
                    // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[1][0] = rightLines[r][0];
                    finalLine[1][1] = rightLines[r][1];
                    finalLine[1][2] = rightLines[r][2];
                    finalLine[1][3] = rightLines[r][3];

                    rslope = slope;
                }
            }

            // 검출된 차선들의 point 담기
            leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
            leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
            rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
            rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

            left.push_back(leftP1);
            left.push_back(leftP2);
            right.push_back(rightP1);
            right.push_back(rightP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
            cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 좌회전 경우
        else if((leftLines.size()==0) && (remainL.size()!=0) && (rightLines.size()!=0)){    // 좌회전 상황
            // 최종차선 초기갑ㅅ 설정
            finalLine[0][0] = remainL[0][0];
            finalLine[0][1] = remainL[0][1];
            finalLine[0][2] = remainL[0][2];
            finalLine[0][3] = remainL[0][3];
            finalLine[1][0] = rightLines[0][0];
            finalLine[1][1] = rightLines[0][1];
            finalLine[1][2] = rightLines[0][2];
            finalLine[1][3] = rightLines[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int l=0; l<remainL.size(); l++){
                if(remainL[l][2]-remainL[l][0] == 0)  slope = 999.9;
                else    slope = (remainL[l][3]-remainL[l][1])/
                         (float)(remainL[l][2]-remainL[l][0]);

                if(slope > lslope){
                    // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[0][0] = remainL[l][0];
                    finalLine[0][1] = remainL[l][1];
                    finalLine[0][2] = remainL[l][2];
                    finalLine[0][3] = remainL[l][3];

                    lslope = slope;
                }
            }
    
            for(int r=0; r<rightLines.size(); r++){
                if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
                else    slope = (rightLines[r][3]-rightLines[r][1])/
                         (float)(rightLines[r][2]-rightLines[r][0]);

                if(slope > rslope){
                    // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[1][0] = rightLines[r][0];
                    finalLine[1][1] = rightLines[r][1];
                    finalLine[1][2] = rightLines[r][2];
                    finalLine[1][3] = rightLines[r][3];

                    rslope = slope;
                }
            }

            // 검출된 차선들의 point 담기
            leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
            leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
            rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
            rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

            left.push_back(leftP1);
            left.push_back(leftP2);
            right.push_back(rightP1);
            right.push_back(rightP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
            cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 우회전 경우
        else if((leftLines.size()!=0) && (rightLines.size()==0) && (remainR.size()!=0)){    // 우회전 상황
            // 최종차선 초기갑ㅅ 설정
            finalLine[0][0] = leftLines[0][0];
            finalLine[0][1] = leftLines[0][1];
            finalLine[0][2] = leftLines[0][2];
            finalLine[0][3] = leftLines[0][3];
            finalLine[1][0] = remainR[0][0];
            finalLine[1][1] = remainR[0][1];
            finalLine[1][2] = remainR[0][2];
            finalLine[1][3] = remainR[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int l=0; l<leftLines.size(); l++){
                if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
                else    slope = (leftLines[l][3]-leftLines[l][1])/
                         (float)(leftLines[l][2]-leftLines[l][0]);

                if(slope < lslope){
                    // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[0][0] = leftLines[l][0];
                    finalLine[0][1] = leftLines[l][1];
                    finalLine[0][2] = leftLines[l][2];
                    finalLine[0][3] = leftLines[l][3];

                    lslope = slope;
                }
            }
    
            for(int r=0; r<remainR.size(); r++){
                if(remainR[r][2]-remainR[r][0] == 0)  slope = 999.9;
                else    slope = (remainR[r][3]-remainR[r][1])/
                         (float)(remainR[r][2]-remainR[r][0]);

                if(slope < rslope){
                    // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[1][0] = remainR[r][0];
                    finalLine[1][1] = remainR[r][1];
                    finalLine[1][2] = remainR[r][2];
                    finalLine[1][3] = remainR[r][3];

                    rslope = slope;
                }
            }

            // 검출된 차선들의 point 담기
            leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
            leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
            rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
            rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

            left.push_back(leftP1);
            left.push_back(leftP2);
            right.push_back(rightP1);
            right.push_back(rightP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
            cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 왼쪽 차선 없을 때
        else if((leftLines.size()==0) && (remainL.size()==0) && (rightLines.size()!=0)){    // 검출된 왼쪽차선이 없을 때
            // 최종차선 초기갑ㅅ 설정
            finalLine[1][0] = rightLines[0][0];
            finalLine[1][1] = rightLines[0][1];
            finalLine[1][2] = rightLines[0][2];
            finalLine[1][3] = rightLines[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int r=0; r<rightLines.size(); r++){
                if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
                else    slope = (rightLines[r][3]-rightLines[r][1])/
                         (float)(rightLines[r][2]-rightLines[r][0]);

                if(slope > rslope){
                    // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[1][0] = rightLines[r][0];
                    finalLine[1][1] = rightLines[r][1];
                    finalLine[1][2] = rightLines[r][2];
                    finalLine[1][3] = rightLines[r][3];

                    rslope = slope;
                }
            }

            // 검출된 오른쪽 차선의 point 담기
            rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
            rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

            cv::Point2f worldR1, worldR2;
            Projection(rightP1,worldR1);
            Projection(rightP2,worldR2);

            cv::Point virP1,virP2;
            virP1.x = worldR1.x - LANEWIDTH;
            virP1.y = worldR1.y;
            virP2.x = worldR2.x - LANEWIDTH;
            virP2.y = worldR2.y;

            cv::Point2f vir_leftP1,vir_leftP2;
            Projection(virP1,vir_leftP1,W2I);
            Projection(virP2,vir_leftP2,W2I);

            right.push_back(rightP1);
            right.push_back(rightP2);
            left.push_back(vir_leftP1);
            left.push_back(vir_leftP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            // cv::Point vp = {(int)((rightP1.x+vir_leftP1.x)/2.0),(int)((rightP1.y+vir_leftP1.y)/2.0)};
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, vir_leftP1, vir_leftP2, GREEN, 2, 8);
            cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 오른쪽 차선 없을 때
        else if((leftLines.size()!=0) && (rightLines.size()==0) && (remainR.size()==0)){    // 검출된 오른쪽차선이 없을 때
            // 최종차선 초기갑ㅅ 설정
            finalLine[0][0] = leftLines[0][0];
            finalLine[0][1] = leftLines[0][1];
            finalLine[0][2] = leftLines[0][2];
            finalLine[0][3] = leftLines[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int l=0; l<leftLines.size(); l++){
                if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
                else    slope = (leftLines[l][3]-leftLines[l][1])/
                         (float)(leftLines[l][2]-leftLines[l][0]);

                if(slope < lslope){
                    // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[0][0] = leftLines[l][0];
                    finalLine[0][1] = leftLines[l][1];
                    finalLine[0][2] = leftLines[l][2];
                    finalLine[0][3] = leftLines[l][3];

                    lslope = slope;
                }
            }

            // 검출된 오른쪽 차선의 point 담기
            leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
            leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};

            cv::Point2f worldL1, worldL2;
            Projection(leftP1,worldL1);
            Projection(leftP2,worldL2);

            cv::Point virP1,virP2;
            virP1.x = worldL1.x + LANEWIDTH;
            virP1.y = worldL1.y;
            virP2.x = worldL2.x + LANEWIDTH;
            virP2.y = worldL2.y;

            cv::Point2f vir_rightP1,vir_rightP2;
            Projection(virP1,vir_rightP1,W2I);
            Projection(virP2,vir_rightP2,W2I);

            right.push_back(vir_rightP1);
            right.push_back(vir_rightP2);
            left.push_back(leftP1);
            left.push_back(leftP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            // cv::Point vp = {(int)((vir_rightP1.x+leftP1.x)/2.0),(int)((vir_rightP2.y+leftP2.y)/2.0)};
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
            cv::line(img_draw, vir_rightP1, vir_rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 왼쪽차선 없고 우회전 경우
        else if((leftLines.size()==0) && (remainL.size()==0) && (rightLines.size()==0) && (remainR.size()!=0)){    
            // 최종차선 초기갑ㅅ 설정
            finalLine[1][0] = remainR[0][0];
            finalLine[1][1] = remainR[0][1];
            finalLine[1][2] = remainR[0][2];
            finalLine[1][3] = remainR[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int r=0; r<remainR.size(); r++){
                if(remainR[r][2]-remainR[r][0] == 0)  slope = 999.9;
                else    slope = (remainR[r][3]-remainR[r][1])/
                         (float)(remainR[r][2]-remainR[r][0]);

                if(slope < rslope){
                    // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[1][0] = remainR[r][0];
                    finalLine[1][1] = remainR[r][1];
                    finalLine[1][2] = remainR[r][2];
                    finalLine[1][3] = remainR[r][3];

                    rslope = slope;
                }
            }

            // 검출된 오른쪽 차선의 point 담기
            rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
            rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

            cv::Point2f worldR1, worldR2;
            Projection(rightP1,worldR1);
            Projection(rightP2,worldR2);

            cv::Point virP1,virP2;
            virP1.x = worldR1.x - LANEWIDTH;
            virP1.y = worldR1.y;
            virP2.x = worldR2.x - LANEWIDTH;
            virP2.y = worldR2.y;

            cv::Point2f vir_leftP1,vir_leftP2;
            Projection(virP1,vir_leftP1,W2I);
            Projection(virP2,vir_leftP2,W2I);

            right.push_back(rightP1);
            right.push_back(rightP2);
            left.push_back(vir_leftP1);
            left.push_back(vir_leftP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            // cv::Point vp = {(int)((rightP1.x+vir_leftP1.x)/2.0),(int)((rightP1.y+vir_leftP1.y)/2.0)};
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, vir_leftP1, vir_leftP2, GREEN, 2, 8);
            cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        // 오른쪽 차선 없고 좌회전 경우
        else if((leftLines.size()==0) && (remainL.size()!=0) && (rightLines.size()==0) && (remainR.size()==0)){    
            // 최종차선 초기갑ㅅ 설정
            finalLine[0][0] = remainL[0][0];
            finalLine[0][1] = remainL[0][1];
            finalLine[0][2] = remainL[0][2];
            finalLine[0][3] = remainL[0][3];

            float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

            for(int l=0; l<remainL.size(); l++){
                if(remainL[l][2]-remainL[l][0] == 0)  slope = 999.9;
                else    slope = (remainL[l][3]-remainL[l][1])/
                         (float)(remainL[l][2]-remainL[l][0]);

                if(slope > lslope){
                    // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                    finalLine[0][0] = remainL[l][0];
                    finalLine[0][1] = remainL[l][1];
                    finalLine[0][2] = remainL[l][2];
                    finalLine[0][3] = remainL[l][3];

                    lslope = slope;
                }
            }

            // 검출된 오른쪽 차선의 point 담기
            leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
            leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};

            cv::Point2f worldL1, worldL2;
            Projection(leftP1,worldL1);
            Projection(leftP2,worldL2);

            cv::Point virP1,virP2;
            virP1.x = worldL1.x + LANEWIDTH;
            virP1.y = worldL1.y;
            virP2.x = worldL2.x + LANEWIDTH;
            virP2.y = worldL2.y;

            cv::Point2f vir_rightP1,vir_rightP2;
            Projection(virP1,vir_rightP1,W2I);
            Projection(virP2,vir_rightP2,W2I);

            right.push_back(vir_rightP1);
            right.push_back(vir_rightP2);
            left.push_back(leftP1);
            left.push_back(leftP2);
            
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            // cv::Point vp = {(int)((vir_rightP1.x+leftP1.x)/2.0),(int)((vir_rightP2.y+leftP2.y)/2.0)};
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
            lp_old = lp;
            cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
            cv::line(img_draw, vir_rightP1, vir_rightP2, MAGENTA, 2, 8);
            cv::circle(img_draw, lp, 5, BLUE, -1);
    
            cv::Point2f wp;
            Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            wp_old = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp.x - (size.width / 2);
	        word_center.y = lp.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
        else{
            std::cout << "No Line Detected!" << std::endl;
            std::string coord = "(" + std::to_string((int)(wp_old.x*100)) + "," + std::to_string((int)(wp_old.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = lp_old.x - (size.width / 2);
	        word_center.y = lp_old.y + (size.height);
	        cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
    }
    else{
        if(show_trackbar){
            cv::createTrackbar("HoughLines_Threshold", "Find Line",  &houghTH, 179);
        }
        std::vector<cv::Vec2f> lines;

        cv::HoughLines(img_edge, lines, 1, CV_PI/180., houghTH);
        std::vector<cv::Vec2f> rightLines, leftLines;
        float finalLine[2][2];
        cv::Point leftP1, leftP2;
        cv::Point rightP1, rightP2;
        std::vector<cv::Point2f> left,right;

        for(int i=0; i<lines.size(); i++){
            float rho = lines[i][0], theta = lines[i][1];
            
            if((theta < CV_PI/2.)){
                // 왼쪽 차선 검출
                leftLines.push_back(lines[i]);
                // std::cout << "left  :" << x1 << ", " <<y1 << std::endl;
            }
            else if((theta > CV_PI/2.)){
                // 오른쪽 차선 검출
                rightLines.push_back(lines[i]);
                // std::cout << "right  :" << x1 << ", " <<y1 << std::endl;
            }
        }

        if((leftLines.size()!=0) && (rightLines.size()!=0)){
            finalLine[0][0] = leftLines[0][0];
            finalLine[0][1] = leftLines[0][1];
            finalLine[1][0] = rightLines[0][0];
            finalLine[1][1] = rightLines[0][1];
    
            for(int l=0; l<leftLines.size(); l++){
                if(leftLines[l][1] < finalLine[0][1]){
                    // 왼쪽 차선중 가장 수직에 가까운 직선을 대표직선으로
                    finalLine[0][0] = leftLines[l][0];
                    finalLine[0][1] = leftLines[l][1];
                }
            }
    
            for(int r=0; r<rightLines.size(); r++){
                if(rightLines[r][1] > finalLine[0][1]){
                    // 오른쪽 차선중 가장 수직에 가까운 직선을 대표직선으로
                    finalLine[1][0] = rightLines[r][0];
                    finalLine[1][1] = rightLines[r][1];
                }
            }
    
            double left_a = cos(finalLine[0][1]), left_b = sin(finalLine[0][1]);
            double left_x0 = left_a*finalLine[0][0], left_y0 = left_b*finalLine[0][0];
    
            double right_a = cos(finalLine[1][1]), right_b = sin(finalLine[1][1]);
            double right_x0 = right_a*finalLine[1][0], right_y0 = right_b*finalLine[1][0];
    
            leftP1.x = cvRound(left_x0 + 1000*(-left_b));
            leftP1.y = cvRound(left_y0 + 1000*(left_a));
            leftP2.x = cvRound(left_x0 - 1000*(-left_b));
            leftP2.y = cvRound(left_y0 - 1000*(left_a));
    
            rightP1.x = cvRound(right_x0 + 1000*(-right_b));
            rightP1.y = cvRound(right_y0 + 1000*(right_a));
            rightP2.x = cvRound(right_x0 - 1000*(-right_b));
            rightP2.y = cvRound(right_y0 - 1000*(right_a));
    
            left.push_back(leftP1);
            left.push_back(leftP2);
            right.push_back(rightP1);
            right.push_back(rightP2);
    
            cv::Point vp = VanishingPoint(left, right); // 최종 차선으로 소실점 검출
            // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
            float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
            float lhBeta = vp.y - lhAlpha*vp.x;
            int lpx = (LD - lhBeta) / lhAlpha;
            cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
            lp = MovingAverageFilter(lp);   // Lookahead Point 필터링

            cv::line(img_draw, leftP1, leftP2, YELLOW, 2, 8);
            cv::line(img_draw, rightP1, rightP2, GREEN, 2, 8);
            cv::circle(img_draw, lp, 5, RED, -1);
    
            cv::Point2f wp;
            Projection(lp,wp);  // Lookahead Point 실제 월드 좌표로 변환[m]
        }
        else{
            std::cout << "No Line Detected!" << std::endl;
        }
        
    }
    cv::imshow("Find Line", img_draw);
    // video_line << img_draw;   // 저장할 영상 이미지
    cv::waitKey(1);
}

void Final_Line(const cv::Mat& img_edge, std::vector<cv::Point2f>& left, std::vector<cv::Point2f>& right, cv::Scalar Lcolor, cv::Scalar Rcolor){

    std::vector<cv::Vec4i> lines;
    
    cv::HoughLinesP(img_edge, lines, 1, CV_PI/180., houghPTH, minLine, maxGap);

    float cx = img_edge.cols/2.;
    float slope_threshold = 0.5;
    std::vector<cv::Vec4i> rightLines, leftLines, remainR, remainL; // 좌우 차선 분리하여 담을 벡터(선분의 시작좌표(x1,y1),끝좌표(x2,y2) 4개 데이터 들어있음)
    float finalLine[2][4];  // 최종 차선 담을 배열
    cv::Point rightP1, rightP2, leftP1, leftP2;

    for(int i=0; i<lines.size(); i++){
        int x1 = lines[i][0];
        int y1 = lines[i][1];
        int x2 = lines[i][2];
        int y2 = lines[i][3];

        float slope; // 선분 기울기

        if(x2-x1 == 0)  slope = 999.9;  // 분모가 0이면 기울기 거의 무한대
        else    slope = (y2-y1)/(float)(x2-x1);

        if(fabs(slope) > slope_threshold){ // 너무 수평인 직선들 제거
            if((slope>0) && (x1>cx) && (x2>cx)){ //(x1>cx+qx) && (x2>cx+qx)
                // 기울기가 양수이고 화면 오른쪽에 위치하면 우측차선으로 분류
                rightLines.push_back(lines[i]);
            }
            else if((slope<0) && (x1<cx) && (x2<cx)){ //(x1<cx-qx) && (x2<cx-qx)
                // 기울기가 음수이고 화면 왼쪽에 위치하면 좌측차선으로 분류
                leftLines.push_back(lines[i]);
            }
            else if((x1>cx) && (x2>cx)){
                // 기울기는 음수이지만 화면 오른쪽에 위치한 차선
                remainR.push_back(lines[i]);
            }
            else if((x1<cx) && (x2<cx)){
                // 기울기는 양수이지만 화면 왼쪽에 위치한 차선
                remainL.push_back(lines[i]);
            }
        }
    }

    // 좌우 차선 둘 다 잘 검출된 경우(직진)
    if((leftLines.size()!=0) && (rightLines.size()!=0)){    // 좌우 차선이 검출 되었는지 확인
        // 최종차선 초기갑ㅅ 설정
        finalLine[0][0] = leftLines[0][0];
        finalLine[0][1] = leftLines[0][1];
        finalLine[0][2] = leftLines[0][2];
        finalLine[0][3] = leftLines[0][3];
        finalLine[1][0] = rightLines[0][0];
        finalLine[1][1] = rightLines[0][1];
        finalLine[1][2] = rightLines[0][2];
        finalLine[1][3] = rightLines[0][3];

        float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

        for(int l=0; l<leftLines.size(); l++){
            if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
            else    slope = (leftLines[l][3]-leftLines[l][1])/
                     (float)(leftLines[l][2]-leftLines[l][0]);

            if(slope < lslope){
                // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[0][0] = leftLines[l][0];
                finalLine[0][1] = leftLines[l][1];
                finalLine[0][2] = leftLines[l][2];
                finalLine[0][3] = leftLines[l][3];

                lslope = slope;
            }
        }
        for(int r=0; r<rightLines.size(); r++){
            if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
            else    slope = (rightLines[r][3]-rightLines[r][1])/
                     (float)(rightLines[r][2]-rightLines[r][0]);

            if(slope > rslope){
                // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[1][0] = rightLines[r][0];
                finalLine[1][1] = rightLines[r][1];
                finalLine[1][2] = rightLines[r][2];
                finalLine[1][3] = rightLines[r][3];

                rslope = slope;
            }
        }

        // 검출된 차선들의 point 담기
        leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
        leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
        rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
        rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

        left.push_back(leftP1);
        left.push_back(leftP2);
        right.push_back(rightP1);
        right.push_back(rightP2);
    }
    // 좌회전 경우
    else if((leftLines.size()==0) && (remainL.size()!=0) && (rightLines.size()!=0)){    // 좌회전 상황
        // 최종차선 초기갑ㅅ 설정
        finalLine[0][0] = remainL[0][0];
        finalLine[0][1] = remainL[0][1];
        finalLine[0][2] = remainL[0][2];
        finalLine[0][3] = remainL[0][3];
        finalLine[1][0] = rightLines[0][0];
        finalLine[1][1] = rightLines[0][1];
        finalLine[1][2] = rightLines[0][2];
        finalLine[1][3] = rightLines[0][3];

        float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

        for(int l=0; l<remainL.size(); l++){
            if(remainL[l][2]-remainL[l][0] == 0)  slope = 999.9;
            else    slope = (remainL[l][3]-remainL[l][1])/
                     (float)(remainL[l][2]-remainL[l][0]);

            if(slope > lslope){
                // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[0][0] = remainL[l][0];
                finalLine[0][1] = remainL[l][1];
                finalLine[0][2] = remainL[l][2];
                finalLine[0][3] = remainL[l][3];

                lslope = slope;
            }
        }
        for(int r=0; r<rightLines.size(); r++){
            if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
            else    slope = (rightLines[r][3]-rightLines[r][1])/
                     (float)(rightLines[r][2]-rightLines[r][0]);

            if(slope > rslope){
                // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[1][0] = rightLines[r][0];
                finalLine[1][1] = rightLines[r][1];
                finalLine[1][2] = rightLines[r][2];
                finalLine[1][3] = rightLines[r][3];

                rslope = slope;
            }
        }

        // 검출된 차선들의 point 담기
        leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
        leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
        rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
        rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

        left.push_back(leftP1);
        left.push_back(leftP2);
        right.push_back(rightP1);
        right.push_back(rightP2);
    }
    // 우회전 경우
    else if((leftLines.size()!=0) && (rightLines.size()==0) && (remainR.size()!=0)){    // 우회전 상황
        // 최종차선 초기갑ㅅ 설정
        finalLine[0][0] = leftLines[0][0];
        finalLine[0][1] = leftLines[0][1];
        finalLine[0][2] = leftLines[0][2];
        finalLine[0][3] = leftLines[0][3];
        finalLine[1][0] = remainR[0][0];
        finalLine[1][1] = remainR[0][1];
        finalLine[1][2] = remainR[0][2];
        finalLine[1][3] = remainR[0][3];

        float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

        for(int l=0; l<leftLines.size(); l++){
            if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
            else    slope = (leftLines[l][3]-leftLines[l][1])/
                     (float)(leftLines[l][2]-leftLines[l][0]);

            if(slope < lslope){
                // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[0][0] = leftLines[l][0];
                finalLine[0][1] = leftLines[l][1];
                finalLine[0][2] = leftLines[l][2];
                finalLine[0][3] = leftLines[l][3];

                lslope = slope;
            }
        }
        for(int r=0; r<remainR.size(); r++){
            if(remainR[r][2]-remainR[r][0] == 0)  slope = 999.9;
            else    slope = (remainR[r][3]-remainR[r][1])/
                     (float)(remainR[r][2]-remainR[r][0]);

            if(slope < rslope){
                // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[1][0] = remainR[r][0];
                finalLine[1][1] = remainR[r][1];
                finalLine[1][2] = remainR[r][2];
                finalLine[1][3] = remainR[r][3];

                rslope = slope;
            }
        }

        // 검출된 차선들의 point 담기
        leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
        leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};
        rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
        rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

        left.push_back(leftP1);
        left.push_back(leftP2);
        right.push_back(rightP1);
        right.push_back(rightP2);
    }
    // 왼쪽 차선 없을 때
    else if((leftLines.size()==0) && (remainL.size()==0) && (rightLines.size()!=0)){    // 검출된 왼쪽차선이 없을 때
        // 최종차선 초기갑ㅅ 설정
        finalLine[1][0] = rightLines[0][0];
        finalLine[1][1] = rightLines[0][1];
        finalLine[1][2] = rightLines[0][2];
        finalLine[1][3] = rightLines[0][3];

        float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

        for(int r=0; r<rightLines.size(); r++){
            if(rightLines[r][2]-rightLines[r][0] == 0)  slope = 999.9;
            else    slope = (rightLines[r][3]-rightLines[r][1])/
                     (float)(rightLines[r][2]-rightLines[r][0]);

            if(slope > rslope){
                // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[1][0] = rightLines[r][0];
                finalLine[1][1] = rightLines[r][1];
                finalLine[1][2] = rightLines[r][2];
                finalLine[1][3] = rightLines[r][3];

                rslope = slope;
            }
        }

        // 검출된 오른쪽 차선의 point 담기
        rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
        rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

        cv::Point2f worldR1, worldR2;
        Projection(rightP1,worldR1);
        Projection(rightP2,worldR2);

        cv::Point2f virP1,virP2;
        virP1.x = worldR1.x - LANEWIDTH;
        virP1.y = worldR1.y;
        virP2.x = worldR2.x - LANEWIDTH;
        virP2.y = worldR2.y;

        std::cout << "virP1 : " << virP1.x << ", " << virP1.y << std::endl;
        std::cout << "virP2 : " << virP2.x << ", " << virP2.y << std::endl;

        cv::Point2f vir_leftP1,vir_leftP2;
        Projection(virP1,vir_leftP1,W2I);
        Projection(virP2,vir_leftP2,W2I);

        right.push_back(rightP1);
        right.push_back(rightP2);
        left.push_back(vir_leftP1);
        left.push_back(vir_leftP2);
        std::cout << "vir_leftP1 : " << vir_leftP1.x << ", " << vir_leftP1.y << std::endl;
        std::cout << "vir_leftP2 : " << vir_leftP2.x << ", " << vir_leftP2.y << std::endl;
    }
    // 오른쪽 차선 없을 때
    else if((leftLines.size()!=0) && (rightLines.size()==0) && (remainR.size()==0)){    // 검출된 오른쪽차선이 없을 때
        // 최종차선 초기갑ㅅ 설정
        finalLine[0][0] = leftLines[0][0];
        finalLine[0][1] = leftLines[0][1];
        finalLine[0][2] = leftLines[0][2];
        finalLine[0][3] = leftLines[0][3];

        float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

        for(int l=0; l<leftLines.size(); l++){
            if(leftLines[l][2]-leftLines[l][0] == 0)  slope = 999.9;
            else    slope = (leftLines[l][3]-leftLines[l][1])/
                     (float)(leftLines[l][2]-leftLines[l][0]);

            if(slope < lslope){
                // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
                finalLine[0][0] = leftLines[l][0];
                finalLine[0][1] = leftLines[l][1];
                finalLine[0][2] = leftLines[l][2];
                finalLine[0][3] = leftLines[l][3];

                lslope = slope;
            }
        }

        // 검출된 오른쪽 차선의 point 담기
        leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
        leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};

        cv::Point2f worldL1, worldL2;
        Projection(leftP1,worldL1);
        Projection(leftP2,worldL2);

        cv::Point2f virP1,virP2;
        virP1.x = worldL1.x + LANEWIDTH;
        virP1.y = worldL1.y;
        virP2.x = worldL2.x + LANEWIDTH;
        virP2.y = worldL2.y;

        cv::Point2f vir_rightP1,vir_rightP2;
        Projection(virP1,vir_rightP1,W2I);
        Projection(virP2,vir_rightP2,W2I);

        right.push_back(vir_rightP1);
        right.push_back(vir_rightP2);
        left.push_back(leftP1);
        left.push_back(leftP2);
    }
    // // 왼쪽차선 없고 우회전 경우
    // else if((leftLines.size()==0) && (remainL.size()==0) && (rightLines.size()==0) && (remainR.size()!=0)){    
    //     // 최종차선 초기갑ㅅ 설정
    //     finalLine[1][0] = remainR[0][0];
    //     finalLine[1][1] = remainR[0][1];
    //     finalLine[1][2] = remainR[0][2];
    //     finalLine[1][3] = remainR[0][3];

    //     float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

    //     for(int r=0; r<remainR.size(); r++){
    //         if(remainR[r][2]-remainR[r][0] == 0)  slope = 999.9;
    //         else    slope = (remainR[r][3]-remainR[r][1])/
    //                  (float)(remainR[r][2]-remainR[r][0]);

    //         if(slope < rslope){
    //             // 오른쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
    //             finalLine[1][0] = remainR[r][0];
    //             finalLine[1][1] = remainR[r][1];
    //             finalLine[1][2] = remainR[r][2];
    //             finalLine[1][3] = remainR[r][3];

    //             rslope = slope;
    //         }
    //     }

    //     // 검출된 오른쪽 차선의 point 담기
    //     rightP1 = {(int)finalLine[1][0],(int)finalLine[1][1]};
    //     rightP2 = {(int)finalLine[1][2],(int)finalLine[1][3]};

    //     cv::Point2f worldR1, worldR2;
    //     Projection(rightP1,worldR1);
    //     Projection(rightP2,worldR2);

    //     cv::Point virP1,virP2;
    //     virP1.x = worldR1.x - LANEWIDTH;
    //     virP1.y = worldR1.y;
    //     virP2.x = worldR2.x - LANEWIDTH;
    //     virP2.y = worldR2.y;

    //     cv::Point2f vir_leftP1,vir_leftP2;
    //     Projection(virP1,vir_leftP1,W2I);
    //     Projection(virP2,vir_leftP2,W2I);

    //     right.push_back(rightP1);
    //     right.push_back(rightP2);
    //     left.push_back(vir_leftP1);
    //     left.push_back(vir_leftP2);
    // }
    // // 오른쪽 차선 없고 좌회전 경우
    // else if((leftLines.size()==0) && (remainL.size()!=0) && (rightLines.size()==0) && (remainR.size()==0)){    
    //     // 최종차선 초기갑ㅅ 설정
    //     finalLine[0][0] = remainL[0][0];
    //     finalLine[0][1] = remainL[0][1];
    //     finalLine[0][2] = remainL[0][2];
    //     finalLine[0][3] = remainL[0][3];

    //     float slope, rslope=0.0, lslope=0.0;    // 기울기 담을 변수

    //     for(int l=0; l<remainL.size(); l++){
    //         if(remainL[l][2]-remainL[l][0] == 0)  slope = 999.9;
    //         else    slope = (remainL[l][3]-remainL[l][1])/
    //                  (float)(remainL[l][2]-remainL[l][0]);

    //         if(slope > lslope){
    //             // 왼쪽 차선 중 기울기가 가장 수직에 가까운 선분 검출
    //             finalLine[0][0] = remainL[l][0];
    //             finalLine[0][1] = remainL[l][1];
    //             finalLine[0][2] = remainL[l][2];
    //             finalLine[0][3] = remainL[l][3];

    //             lslope = slope;
    //         }
    //     }

    //     // 검출된 오른쪽 차선의 point 담기
    //     leftP1 = {(int)finalLine[0][0],(int)finalLine[0][1]};
    //     leftP2 = {(int)finalLine[0][2],(int)finalLine[0][3]};

    //     cv::Point2f worldL1, worldL2;
    //     Projection(leftP1,worldL1);
    //     Projection(leftP2,worldL2);

    //     cv::Point virP1,virP2;
    //     virP1.x = worldL1.x + LANEWIDTH;
    //     virP1.y = worldL1.y;
    //     virP2.x = worldL2.x + LANEWIDTH;
    //     virP2.y = worldL2.y;

    //     cv::Point2f vir_rightP1,vir_rightP2;
    //     Projection(virP1,vir_rightP1,W2I);
    //     Projection(virP2,vir_rightP2,W2I);

    //     right.push_back(vir_rightP1);
    //     right.push_back(vir_rightP2);
    //     left.push_back(leftP1);
    //     left.push_back(leftP2);
    // }
    // else{
    //     // std::cout << "No Line Detected!" << std::endl;
    // }

    if((right.size()!=0) && (left.size()!=0)){
        // 차선 그리기
        cv::line(img_comb, left[0], left[1], Lcolor, 2, 8);
        cv::line(img_comb, right[0], right[1], Rcolor, 2, 8);
    }
}

void Line_detect(const cv::Mat& img_edge, const cv::Mat& img_draw, bool show_trackbar){
    img_comb = img_draw.clone();
    cv::Mat img_roi1, img_roi2, img_roi3;
    std::vector<cv::Point2f> L1, L2, L3, R1, R2, R3;
    cv::Point cp1, vp1, vp2, vp3;
    cv::Point2f s1, s2, s3;
    int cx = img_comb.cols/2.;
    int qx = cx/2.;
    setROIGray(img_edge, img_roi1, roi1);
    // setROIGray(img_edge, img_roi2, roi2);
    // setROIGray(img_edge, img_roi3, roi3);
    bool sec1_okay = 0, sec2_okay = 0, sec3_okay = 0;

    if(show_trackbar){
        cv::createTrackbar("HoughLinesP_Threshold", "Line_detect", &houghPTH, 179);
        cv::createTrackbar("minLineLength",         "Line_detect",  &minLine, 179);
        cv::createTrackbar("maxLineGap",            "Line_detect",   &maxGap, 179);
    }

    Final_Line(img_roi1, L1, R1);
    // Final_Line(img_roi2, L2, R2, MAGENTA, GREEN);
    // Final_Line(img_roi3, L3, R3, PURPLE, YELLOW);

    if((L1.size()!=0) && (R1.size()!=0)){
        float L1Alpha, R1Alpha;
        float L1p, R1p;
        if((L1[1].x - L1[0].x) == 0) L1p = L1[0].x;
        else{
            L1Alpha = (L1[1].y - L1[0].y)/(float)(L1[1].x - L1[0].x);
            float L1Beta = L1[1].y - L1Alpha*L1[1].x;
            L1p = (SECTION1 - L1Beta)/L1Alpha;
        } 
        if((R1[1].x - R1[0].x) == 0) R1p = R1[0].x;
        else{
            R1Alpha = (R1[1].y - R1[0].y)/(float)(R1[1].x - R1[0].x);
            float R1Beta = R1[1].y - R1Alpha*R1[1].x;
            R1p = (SECTION1 - R1Beta)/R1Alpha;
        } 
        // std::cout << "L1p : " << L1p << std::endl;
        // std::cout << "R1p : " << R1p << std::endl;

        // std::vector<cv::Point2f> WL1, WR1;
        // Projection(L1,WL1);
        // Projection(R1,WR1);
        // cv::Point2f WC1, PC1;
        // WC1.x = ((WL1[0].x + LANEWIDTH)+(WL1[1].x + LANEWIDTH)+(WR1[0].x - LANEWIDTH)+(WR1[1].x - LANEWIDTH))/4.0;
        // WC1.y = (WL1[0].y+WL1[1].y+WR1[0].y+WR1[1].y)/4.0;
        // Projection(WC1,PC1,W2I);
        // cp1 = {(int)PC1.x, SECTION1};
        cv::Point bottom = MovingAverageFilter({L1p,R1p},BOTTOM_buf);
        L1p = bottom.x;
        R1p = bottom.y;

        std::string coord = "L " + std::to_string((int)L1p) + ", R " + std::to_string((int)R1p);
	    cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	    cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	    word_center.x = (img_comb.cols/2) - (size.width / 2);
	    word_center.y = (img_comb.rows-30) + (size.height);
	    cv::putText(img_comb, coord, word_center, font, fontScale, BLACK, thickness, 8);
        
        if((L1p < cx) && (R1p > cx)){
            cp1 = {(int)((L1p+R1p)/2), SECTION1};
            cp1 = MovingAverageFilter(cp1);   // 필터링
            vp1 = VanishingPoint(L1, R1); // 최종 차선으로 소실점 검출
            // vp1 = MovingAverageFilter(vp1,SEC1_buf);   // 필터링
            sec1_okay = 1;
        }
    } 
    // if((L2.size()!=0) && (R2.size()!=0)){
    //     vp2 = VanishingPoint(L2, R2); // 최종 차선으로 소실점 검출
    //     // vp2 = MovingAverageFilter(vp2,SEC2_buf);   // 필터링
    //     sec2_okay = 1;
    // } 
    // if((L3.size()!=0) && (R3.size()!=0)){
    //     vp3 = VanishingPoint(L3, R3); // 최종 차선으로 소실점 검출
    //     // vp3 = MovingAverageFilter(vp3,SEC3_buf);   // 필터링
    //     sec3_okay = 1;
    // } 
    
    if(sec1_okay){
        if((vp1.x - cp1.x) == 0) s1 = {(float)cp1.x, SECTION2};
        else{
            float S1Alpha = (vp1.y - cp1.y)/(float)(vp1.x - cp1.x);
            float S1Beta = vp1.y - S1Alpha*vp1.x;
            float S1p = (SECTION2 - S1Beta)/S1Alpha;
            s1 = {S1p, SECTION2};
        } 
        if((s1.x>0) && (s1.x<img_comb.cols)){
            s1 = MovingAverageFilter(s1,SEC1_buf);   // 필터링
            // std::cout << "cp1 : " << cp1.x << ", " << cp1.y << std::endl;
            // std::cout << "s1 : " << s1.x << ", " << s1.y << std::endl;
            old_s1 = s1;
            // cv::line(img_comb, cp1, s1, BLACK, 2, 8);
            // cv::circle(img_comb, cp1, 5, PURPLE, -1);
            cv::circle(img_comb, s1, 5, BLACK, -1);
            cv::circle(img_comb, s1, 5, RED, -1);
            cv::Point2f wp;
            Projection(s1,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
            old_wp = wp;
            std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	        cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	        cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	        word_center.x = s1.x - (size.width / 2);
	        word_center.y = s1.y + (size.height);
	        cv::putText(img_comb, coord, word_center, font, fontScale, BLACK, thickness, 8);
        }
    }
    else{
        cv::circle(img_comb, old_s1, 5, RED, -1);
        std::string coord = "(" + std::to_string((int)(old_wp.x*100)) + "," + std::to_string((int)(old_wp.y*100)) + ") cm";
	    cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	    cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	    word_center.x = old_s1.x - (size.width / 2);
	    word_center.y = old_s1.y + (size.height);
	    cv::putText(img_comb, coord, word_center, font, fontScale, BLACK, thickness, 8);
    }
    
    // if(sec1_okay && sec2_okay){
    //     if((vp2.x - vp1.x) == 0) s2 = {(float)vp1.x, SECTION3};
    //     else{
    //         float S2Alpha = (vp2.y - vp1.y)/(float)(vp2.x - vp1.x);
    //         float S2Beta = vp2.y - S2Alpha*vp2.x;
    //         float S2p = (SECTION3 - S2Beta)/S2Alpha;
    //         s2 = {S2p, SECTION3};
    //     } 
    //     if((s2.x>0) && (s2.x<img_comb.cols)){
    //         s2 = MovingAverageFilter(s2,SEC2_buf);   // 필터링
    //         std::cout << "s2 : " << s2.x << ", " << s2.y << std::endl;
    //         cv::line(img_comb, s1, s2, GRAY, 2, 8);
    //         cv::circle(img_comb, s2, 5, GRAY, -1);
    //     }
    // }
    
    // if(sec2_okay && sec3_okay){
    //     if((vp3.x - vp2.x) == 0) s3 = {(float)vp2.x, SECTIONEND};
    //     else{
    //         float S3Alpha = (vp3.y - vp2.y)/(float)(vp3.x - vp2.x);
    //         float S3Beta = vp3.y - S3Alpha*vp3.x;
    //         float S3p = (SECTIONEND - S3Beta)/S3Alpha;
    //         s3 = {S3p, SECTIONEND};
    //     } 
    //     if((s3.x>0) && (s3.x<img_comb.cols)){
    //         s3 = MovingAverageFilter(s3,SEC3_buf);   // 필터링
    //         std::cout << "s3 : " << s3.x << ", " << s3.y << std::endl;
    //         cv::line(img_comb, s2, s3, WHITE, 2, 8);
    //         cv::circle(img_comb, s3, 5, WHITE, -1);
    //     }
    // }
    
    // // 검출된 소실점과 화면 하단 가운데 점을 연결한 선분에서 Lookahead Distance에 가까운 x좌표 계산
    // float lhAlpha = (vp.y- img_draw.rows)/(float)(vp.x-img_draw.cols/2);
    // float lhBeta = vp.y - lhAlpha*vp.x;
    // int lpx = (LD - lhBeta) / lhAlpha;
    // cv::Point lp = {lpx, LD};       // 계산된 Lookahead Point
    // lp = MovingAverageFilter(lp);   // Lookahead Point 필터링
    // lp_old = lp;
    // cv::line(img_draw, leftP1, leftP2, GREEN, 2, 8);
    // cv::line(img_draw, rightP1, rightP2, MAGENTA, 2, 8);
    // cv::circle(img_draw, lp, 5, BLUE, -1);
    // cv::Point2f wp;
    // Projection(lp,wp); // Lookahead Point 실제 월드 좌표로 변환[m]
    // wp_old = wp;
    // std::string coord = "(" + std::to_string((int)(wp.x*100)) + "," + std::to_string((int)(wp.y*100)) + ") cm";
	// cv::Size size = cv::getTextSize(coord, font, fontScale, thickness, &baseLine);	//text사이즈계산 함수
	// cv::Point word_center;	//text의 중심좌표를 word좌표와 일치시키기위한 계산식
	// word_center.x = lp.x - (size.width / 2);
	// word_center.y = lp.y + (size.height);
	// cv::putText(img_draw, coord, word_center, font, fontScale, BLACK, thickness, 8);

    cv::imshow("Line_detect", img_comb);
    video_line << img_comb;   // 저장할 영상 이미지
    cv::waitKey(1);

}

cv::Point2f VanishingPoint(const std::vector<cv::Point2f>& leftLine, const std::vector<cv::Point2f>& rightLine){
    cv::Point2f vp;
    float rAlpha = (rightLine[1].y-rightLine[0].y)/(float)(rightLine[1].x-rightLine[0].x);
    float rBeta = rightLine[1].y - rAlpha*rightLine[1].x;
    float lAlpha = (leftLine[1].y-leftLine[0].y)/(float)(leftLine[1].x-leftLine[0].x);
    float lBeta = leftLine[1].y - lAlpha*leftLine[1].x;

    vp.x = ((rBeta-lBeta)/(lAlpha-rAlpha));
    vp.y = (lAlpha*vp.x + lBeta);

    return vp;
}

cv::Point2f MovingAverageFilter(const cv::Point& array, std::vector<cv::Point2f>& buf, size_t filter_size){
    float sumx = 0.0, sumy = 0.0;
    float avgx, avgy;

    if(buf.size()==0){
        // 벡터크기가 0이면(배열이 비었으면) --> 첫 데이터로 다 채우기
        for(int i=0; i<filter_size; i++){
            buf.push_back(array);
        }
        return array;
    }
    else{
        buf.push_back(array);
        for(int i=0; i<filter_size; i++){
            sumx += buf[buf.size()-(1+i)].x;
            sumy += buf[buf.size()-(1+i)].y;
        }
        avgx = sumx / (float)filter_size;
        avgy = sumy / (float)filter_size;
        return cv::Point2f(avgx,avgy);
    }

}

void Find_Contours(const cv::Mat& img_edge, cv::Mat& img_contour){
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> poly;

    cv::findContours(img_edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    for(int i=0; i<contours.size(); i++){
        std::cout << " contour!!" << std::endl;
        cv::approxPolyDP(contours[i], poly, 1, true);
        std::cout << " poly>>" << poly.size() << std::endl;
        if(poly.size() == 4){
            for(int p=0; p<4; p++){
                cv::line(img_contour, poly[p], poly[(p+1)%4], RED, 2);
            }
        }
        //cv::drawContours(img_contour, contours, i, BLUE, 2, 8, hierarchy, 0);
    }
    cv::imshow("Contours", img_contour);
    cv::waitKey(1);
}

void Histogram(const cv::Mat& img_binary){
    int nH = img_binary.rows;
    int nW = img_binary.cols; 

    cv::MatND histo;
    const int* channel_numbers = {0};
    float channel_range[] = {0.0, 255.0};
    const float* channel_ranges = channel_range;
    int number_bins = 255;

    cv::calcHist(&img_binary, 1, channel_numbers, cv::Mat(), histo, 1, &number_bins, &channel_ranges);
    
    int histW = img_binary.cols, histH = img_binary.rows;
    int binW = cvRound((double)histW/number_bins);

    cv::Mat histImg(histH, histW, CV_8UC1, cv::Scalar(0,0,0));
    normalize(histo, histo, 0, histImg.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < number_bins; i++)
	{
        cv::line(histImg, cv::Point(binW*(i-1), histH - cvRound(histo.at<float>(i-1))), cv::Point(binW*i, histH-cvRound(histo.at<float>(i))), cv::Scalar(255,0,0),1,8,0);
		// if (cvRound(histo.at<float>(i)) == 0) continue;
		// rectangle(histImg, cv::Point(binW * i, histH),
		// 	cv::Point(binW * (i + 1), histH - cvRound(histo.at<float>(i))),
		// 	cv::Scalar(0, 0, 0), -1, 8, 0);
	}
	// for (int i = 1; i < number_bins; i++)
	// {
	// 	cv::line(histImg, cv::Point(binW*(i - 1), histH - cvRound(histo.at<float>(i - 1))),
	// 		cv::Point(binW*(i), histH - cvRound(histo.at<float>(i))),
	// 		cv::Scalar(255, 0, 0), 2, 8, 0);
	// }
    // cv::Mat img_histo(cv::Size(nW,nH), CV_8UC1);
    // int histo[nW];
    // double ratio[nW];
    // int total = 0;

    // for(int i=0; i<nH; i++){
    //     for(int j=0; j<nW; j++){
    //         if(img_binary.at<uchar>(i,j) == 0){
    //             histo[j]++;
    //             total++;
    //         }
    //     }
    // }

    // for(int h=0; h<nW; h++){
    //     ratio[h] = histo[h]/(double)total;
    // }

    // for(int i=0; i<nH; i++){
    //     for(int j=0; j<nW; j++){
    //         if(i>(nH-(ratio[j]*nH))){
    //             img_histo.at<uchar>(i,j) = 255;
    //         }
    //         else{
    //             img_histo.at<uchar>(i,j) = 0;
    //         }
    //     }
    // }

    cv::imshow("Histogram", histImg);
    cv::waitKey(0);
}

void makeobjectPoints(std::vector<cv::Point3f>& obj, const int mode, const bool show_detail){
    float init_x = 0.0, init_y = 0.0, init_z = 0.0;
    obj.clear();

    std::cout << " Make Obj Start! " << std::endl;

    // Chessboard함수에서 찾은 시작점 기준 월드 좌표 원점 잡고,
    // 실제 a4 프린트 된 chessboard 격자 사이즈에 맞게 월드 좌표 벡터들 생성
    for(int w=0; w<CHESS_COLS; w++){
        for(int h=0; h<CHESS_ROWS; h++){
            float x = init_x + (w*CHESSBOARDGRID);
            float y = init_y + (h*CHESSBOARDGRID);
            float z = 0.0;
            obj.push_back({x, y, z});
        }
    }
    // if((mode == 1) || (mode == 2)){
    //     for(int w=0; w<CHESS_COLS; w++){
    //         for(int h=0; h<CHESS_ROWS; h++){
    //             float x = init_x + (w*CHESSBOARDGRID);
    //             float y = init_y + (h*CHESSBOARDGRID);
    //             float z = 0.0;
    //             obj.push_back({x, y, z});
    //         }
    //     }
    // }
    // else if((mode == 3) || (mode == 4)){
    //     std::cout << " if 3,4" << std::endl;
    //     for(int h=0; h<CHESS_ROWS; h++){
    //         for(int w=0; w<CHESS_COLS; w++){
    //             float x = init_x + (w*CHESSBOARDGRID);
    //             float y = init_y - (h*CHESSBOARDGRID);
    //             float z = 0.0;
    //             obj.push_back({x, y, z});
    //         }
    //     }
    // }

    // if(mode == 1){
    //     std::cout << " 세로 빨주노초파남 " << std::endl;
    // }
    // else if(mode == 2){
    //     std::cout << " 세로 남파초노주빨 " << std::endl;
    //     std::reverse(obj.begin(), obj.end());
    // }
    // else if(mode == 3){
    //     std::cout << " 가로 빨주노초파남 " << std::endl;
    // }
    // else if(mode == 4){
    //     std::cout << " 가로 남파초노주빨 " << std::endl;
    //     std::reverse(obj.begin(), obj.end());
    // }
    // else{
    //     std::cout << " XXX Make Obj ERROR! XXX " << std::endl;
    // }

    if(show_detail){
        for(int i=0; i<obj.size(); i++){
            std::cout << i << ") " << "x : " << obj[i].x << " y : " << obj[i].y << " z : " << obj[i].z << std::endl;
        }
        std::cout << " Make Obj size! " << obj.size() << std::endl;
    }
    std::cout << " Make Obj Finish! " << std::endl;
}

void ChessBoard(const cv::Mat& img,  std::vector<cv::Point2f>& corners , bool show_board){
    cv::Mat img_board = img.clone();
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    bool patternfound = cv::findChessboardCorners(img_gray, patternsize, corners);

    if(patternfound)
      cv::cornerSubPix(img_gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    cv::drawChessboardCorners(img_board, patternsize, cv::Mat(corners), patternfound);

    if(show_board){
        if(patternfound) std::cout << " Pattern Found!!! " << std::endl;
        else std::cout << " xxx No Pattern xxx " << std::endl;
    
        std::cout << " ChessBoard Start! " << std::endl;
        for(int i=0; i<corners.size();i++){
            std::cout << i << ") ";
            std::cout << " x : "<< corners[i].x;
            std::cout << " y : "<< corners[i].y << std::endl;
        }
        std::cout << " ChessBoard Finish! " << std::endl;

        cv::imshow("ChessBoard", img_board);
        cv::waitKey(0);
    }
}

void drawXYZ(const cv::Mat& img, const camParam& camera, const bool mode = 0, double x, double y, double z){
    std::cout << " Draw XYZ! " << std::endl;
    cv::Mat img_draw = img.clone();
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::Point3f> obj_pts;

    obj_pts.push_back(cv::Point3f(0, 0, 0));
    obj_pts.push_back(cv::Point3f(x, 0, 0));
    obj_pts.push_back(cv::Point3f(0, y, 0));
    obj_pts.push_back(cv::Point3f(0, 0, z));

    projectPoints(
                obj_pts,
                camera.Rotation, camera.Translation,
                camera.cameraMatrix, camera.distCoeffs,
                img_pts);

    cv::circle(img_draw, img_pts.at(0), 10, RED);
    cv::line(img_draw, img_pts.at(0), img_pts.at(1), RED, 5, 8, 0);   // x축
    cv::line(img_draw, img_pts.at(0), img_pts.at(2), GREEN, 5, 8, 0);   // y축
    cv::line(img_draw, img_pts.at(0), img_pts.at(3), BLUE, 5, 8, 0);   // z축
    
    cv::imshow("draw XYZ", img_draw);
    if(mode == 0) cv::waitKey(1);
    else cv::waitKey(0);
}

void ExtrinsicCalibration(const cv::Mat& img, camParam& camera, bool show_cali){
    cv::Mat img_line = img.clone();
    std::vector<cv::Point2f> imagePoints, drawPoints;
    std::vector<cv::Point3f> objectPoints;

    ChessBoard(img, imagePoints, show_cali);   // imagePoints 찾아주는 함수
    if(imagePoints.size() != CHESS_ROWS*CHESS_COLS)
        return;

    int mode = 0;
    if((imagePoints[0].x < imagePoints[47].x) && (imagePoints[0].y > imagePoints[47].y)) mode = 1;
    else if((imagePoints[0].x > imagePoints[47].x) && (imagePoints[0].y < imagePoints[47].y)) mode = 2;
    else if((imagePoints[0].x < imagePoints[47].x) && (imagePoints[0].y < imagePoints[47].y)) mode = 3;
    else if((imagePoints[0].x > imagePoints[47].x) && (imagePoints[0].y > imagePoints[47].y)) mode = 4;
    makeobjectPoints(objectPoints, mode, show_cali);

    std::cout << " Extrin Cali Start! " << std::endl;

    // Rotation, Translation Matrix
    cv::Mat Rotation, Translation;
    cv::solvePnP(objectPoints, imagePoints, 
                 cameraMatrix, distCoeffs, 
                 Rotation, Translation);
    cv::Mat R, T;
    cv::Rodrigues(Rotation, R);
    cv::Mat R_inv = R.inv();
    T = Translation;

    camera.cameraMatrix = cameraMatrix;
    camera.distCoeffs   = distCoeffs;
    camera.Rotation     = Rotation;
    camera.Translation  = Translation;

    cv::Mat Cam_pos = -R_inv * Translation;
    double* p = (double*)Cam_pos.data;
    camera.X = p[0], camera.Y=p[1], camera.Z=p[2];    // 카메라 위치 x,y,z

    double unit_z[] = {0., 0., 1.};
    cv::Mat Zc(3, 1, CV_64FC1, unit_z);
    cv::Mat Zw = R_inv * Zc;
    double* zw = (double*)Zw.data;

    camera.pan = atan2(zw[1], zw[0]) - CV_PI/2;     //카메라 좌우 회전각 (왼쪽 +, 오른쪽 -)
    camera.tilt = atan2(zw[2], sqrt(zw[0]*zw[0] + zw[1]*zw[1])); // 카메라 상하 회전각 (위쪽 +, 아래쪽 -)

    double unit_x[] = {1., 0., 0.};
    cv::Mat Xc(3, 1, CV_64FC1, unit_x);
    cv::Mat Xw = R_inv * Xc;
    double* xw = (double*)Xw.data;
    double xpan[] = {cos(camera.pan), sin(camera.pan), 0};

    // double roll = acos(xw[0]*xpan[0] + xw[1]*xpan[1] + xw[2]*xpan[2]); // 카메라 광학축 기준 회전각 (카메라와 같은 방향을 바라볼 때, 시계 방향 +, 반시계 방향 -)
    // if(xw[2]<0) roll = -roll;

    std::cout << " Extrin Cali Finish! " << std::endl;

    if(show_cali){
        std::cout << "----cameraMatrix----" << std::endl;
        std::cout << cameraMatrix << std::endl;
        std::cout << "----distCoeffs----" << std::endl;
        std::cout << distCoeffs << std::endl;
        std::cout << "----Rotation Vector----" << std::endl;
        std::cout << Rotation << std::endl;
        std::cout << "----Translation Vector----" << std::endl;
        std::cout << Translation << std::endl;
    
        std::cout << "Camera Extrin Parameter >>" << std::endl;
        std::cout << " X : "    << camera.X     << std::endl;
        std::cout << " Y : "    << camera.Y     << std::endl;
        std::cout << " Z : "    << camera.Z     << std::endl;
        std::cout << " pan : "  << camera.pan   << std::endl;
        std::cout << " tilt : " << camera.tilt  << std::endl;
    }
    drawXYZ(img_line, camera, show_cali);    
}

cv::Point2f transformPoint(const cv::Point2f& cur, const cv::Mat& T){
    cv::Point2f tPoint;

    tPoint.x = cur.x * T.at<double>(0,0) 
             + cur.y * T.at<double>(0,1) 
             + T.at<double>(0,2);
    tPoint.y = cur.x * T.at<double>(1,0) 
             + cur.y * T.at<double>(1,1) 
             + T.at<double>(1,2);
    float z = cur.x * T.at<double>(2,0) 
            + cur.y * T.at<double>(2,1) 
            + T.at<double>(2,2);       

    tPoint.x /= z;      
    tPoint.y /= z;      

    return tPoint;
}

void Projection(const cv::Point2f& src, cv::Point2f& dst, bool direction){
    cv::Mat img_warp;
    cv::Point2f p;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> objectPoints;

    // imagePoints.push_back(cv::Point2f(266,186));
    // imagePoints.push_back(cv::Point2f(205,300));
    // imagePoints.push_back(cv::Point2f(553,305));
    // imagePoints.push_back(cv::Point2f(487,188));

    // objectPoints.push_back(cv::Point2f(-0.15,1.50));
    // objectPoints.push_back(cv::Point2f(-0.15,1.00));
    // objectPoints.push_back(cv::Point2f( 0.15,1.00));
    // objectPoints.push_back(cv::Point2f( 0.15,1.50));

    imagePoints.push_back(cv::Point2f(86,479));
    imagePoints.push_back(cv::Point2f(384,479));
    imagePoints.push_back(cv::Point2f(385,201));
    imagePoints.push_back(cv::Point2f(235,198));

    objectPoints.push_back(cv::Point2f(-0.289,1.02));
    objectPoints.push_back(cv::Point2f( 0.011,1.02));
    objectPoints.push_back(cv::Point2f( 0.011,1.92));
    objectPoints.push_back(cv::Point2f(-0.289,1.92));
    

    cv::Mat img2World = cv::getPerspectiveTransform(imagePoints, objectPoints);
    cv::Mat world2Image = img2World.inv();

    if(direction){
        p = transformPoint(src, img2World); 
        // std::cout << "Real : " << p.x << ", " << p.y << std::endl;
    }        
    else{
        p = transformPoint(src, world2Image);
        // std::cout << "IMG : " << p.x << ", " << p.y << std::endl;
    }            
    dst = p;
}

void Projection(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst, bool direction){
    cv::Mat img_warp;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> objectPoints;

    imagePoints.push_back(cv::Point2f(86,479));
    imagePoints.push_back(cv::Point2f(384,479));
    imagePoints.push_back(cv::Point2f(385,201));
    imagePoints.push_back(cv::Point2f(235,198));

    objectPoints.push_back(cv::Point2f(-0.289,1.02));
    objectPoints.push_back(cv::Point2f( 0.011,1.02));
    objectPoints.push_back(cv::Point2f( 0.011,1.92));
    objectPoints.push_back(cv::Point2f(-0.289,1.92));

    cv::Mat img2World = cv::getPerspectiveTransform(imagePoints, objectPoints);
    cv::Mat world2Image = img2World.inv();

    for(int i=0; i<src.size(); i++){
        cv::Point2f p;
        if(direction)   p = transformPoint(src[i], img2World);      
        else            p = transformPoint(src[i], world2Image);
        dst.push_back(p);
        // std::cout << "Real [" << i << "] (" << p.x << ", " << p.y << std::endl;
    }
}

void BirdEyeView(const cv::Mat& src, cv::Mat& dst){
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> objectPoints;

    imagePoints.push_back(cv::Point2f(252,285));
    imagePoints.push_back(cv::Point2f(59,575));
    imagePoints.push_back(cv::Point2f(1007,507));
    imagePoints.push_back(cv::Point2f(813,269));

    objectPoints.push_back(cv::Point2f(0,0));
    objectPoints.push_back(cv::Point2f(0,src.rows));
    objectPoints.push_back(cv::Point2f(src.cols,src.rows));
    objectPoints.push_back(cv::Point2f(src.cols,0));

    cv::Mat img2World = cv::getPerspectiveTransform(imagePoints, objectPoints);
    cv::Mat world2Image = img2World.inv();

    cv::warpPerspective(src, dst, img2World,cv::Size(src.cols,src.rows));

    cv::imshow("BirdEye View", dst);
    cv::waitKey(1);
}

void BirdEyeView(const cv::Mat& src, cv::Mat& dst, const std::vector<cv::Point2f>& imagePoints){
    std::vector<cv::Point2f> objectPoints;

    objectPoints.push_back(cv::Point2f(0,0));
    objectPoints.push_back(cv::Point2f(0,src.rows));
    objectPoints.push_back(cv::Point2f(src.cols,src.rows));
    objectPoints.push_back(cv::Point2f(src.cols,0));

    cv::Mat img2World = cv::getPerspectiveTransform(imagePoints, objectPoints);
    cv::Mat world2Image = img2World.inv();

    cv::warpPerspective(src, dst, img2World,cv::Size(src.cols,src.rows));

    cv::imshow("BirdEye View", dst);
    cv::waitKey(1);
}

