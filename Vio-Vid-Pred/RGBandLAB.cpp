#include "stdafx.h"

#include "RGBandLAB.h"

using namespace std;
using namespace cv;

RGBandLAB::RGBandLAB(void){

	m_frame.release();
	m_color_feature.clear();
	m_file_name.clear();

}

RGBandLAB::~RGBandLAB(void){

	m_frame.release();
	m_color_feature.clear();
	m_file_name.clear();

}

bool RGBandLAB::cal_brglab_feature(const cv::Mat& frame, std::vector<double>& bl_feature){

	if(frame.empty()){
		std::cerr << "Sorry, the frame is empty! " << std::endl;
		return false;
	}

	m_frame = frame;

	cv::Mat color_mean = cv::Mat::zeros(3, 1, CV_64FC1);
	cv::Mat color_sdev  = cv::Mat::zeros(3, 1, CV_64FC1);;
	cv::Size frame_size = m_frame.size();

	cv::meanStdDev(m_frame, color_mean, color_sdev);
	
	m_color_feature.push_back(color_mean.at<double>(0,0)); //Blue
	m_color_feature.push_back(color_mean.at<double>(1,0)); //Green
	m_color_feature.push_back(color_mean.at<double>(2,0)); //Red
	m_color_feature.push_back(color_sdev.at<double>(0,0));	//Blue
	m_color_feature.push_back(color_sdev.at<double>(1,0));	//Green
	m_color_feature.push_back(color_sdev.at<double>(2,0));	//Red

	cv::Mat frame_lab = cv::Mat(frame_size, CV_64FC3);
	cv::cvtColor(m_frame, frame_lab, CV_BGR2Lab);

	cv::meanStdDev(frame_lab, color_mean, color_sdev);

	m_color_feature.push_back(color_mean.at<double>(0,0)); //L
	m_color_feature.push_back(color_mean.at<double>(1,0)); //a
	m_color_feature.push_back(color_mean.at<double>(2,0)); //b
	m_color_feature.push_back(color_sdev.at<double>(0,0));	//L
	m_color_feature.push_back(color_sdev.at<double>(1,0));	//a
	m_color_feature.push_back(color_sdev.at<double>(2,0));	//b

	bl_feature = m_color_feature;

	m_color_feature.clear();

	return true;
}

bool RGBandLAB::write(const std::string file_name){
	
	m_file_name = file_name;

	std::ofstream ofs(m_file_name, ios::out);

	if(ofs.fail()){
		std::cerr << "Sorry, failed to open " << m_file_name << "! " << std::endl;
		return false;
	}

	for(unsigned int i=0; i<m_color_feature.size(); i++){
		ofs << m_color_feature[i] << " ";
	}
	ofs << std::endl;
	ofs.close();

	return true;

}