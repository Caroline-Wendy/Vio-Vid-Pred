// Vio-Vid-Pred.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <iostream>
#include <vector>
#include <string>
#include <ctime>

#include <opencv.hpp>

#include "MotionAngle.h"
#include "MotionFlow.h"
#include "RGBandLAB.h"

using namespace std;
using namespace cv;

bool write(const std::string file_name, const vector<double> feature){

	std::ofstream ofs(file_name, ios::app);

	if(ofs.fail()){
		std::cerr << "Sorry, failed to open " << file_name << "! " << std::endl;
		return false;
	}

	for(unsigned int i=0; i<feature.size(); i++){
		ofs << feature[i] << " ";
	}
	ofs << endl;

	ofs.close();

	return true;

}

int main(){

	const std::string file_name = "twinkle_0.59.avi";
	const std::string feature_name = "feature.txt";

	vector<double> ma_feature; //角度向量3
	vector<double> mf_feature; //运动向量3
	vector<double> bl_feature; //颜色向量12

	vector<double> feature;

	cv::VideoCapture capture(file_name);
	if(!capture.isOpened()){
		std::cerr << "Sorry, failed to open " << file_name << "! " << std::endl;
		return -1;
	}

	cv::Mat frame;

	MotionAngle cMA;
	MotionFlow cMF;
	RGBandLAB cRL;

	capture>>frame;

	cv::Mat m_buf[4];

	unsigned long int frame_num;
	frame_num = static_cast<unsigned long int>(capture.get(CV_CAP_PROP_FRAME_COUNT));

	cv::Size key_frame_size = cv::Size(640,360);
	cv::Mat key_frame = cv::Mat::Mat(key_frame_size, CV_32FC3); /*存储缩放帧*/

	while(1){

		capture >> frame;
		if(frame.empty()) break;
		cv::resize(frame, key_frame, key_frame.size(), 0, 0, CV_INTER_AREA);
		cMA.motion_angle_feature(key_frame, ma_feature, true);
		cMF.cal_motion_vector(key_frame, mf_feature, true);
		cRL.cal_brglab_feature(key_frame, bl_feature);

		//write(feature_name, feature);
	}

	std::cout << "The program is over! Thank you! " << std::endl;

}


