#pragma once

#include <iostream>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>

#include <opencv.hpp>

using namespace std;
using namespace cv;

class RGBandLAB{

public:
	RGBandLAB(void);
	~RGBandLAB(void);

public:
	bool RGBandLAB::cal_brglab_feature(const cv::Mat& frame, std::vector<double>& bl_feature);
	bool write(const std::string file_name);

private:
	cv::Mat m_frame; // ‰»ÎÕºœÒ
	std::vector<double> m_color_feature; //1~6, BGR; 7~12, Lab
	std::string m_file_name;

};

