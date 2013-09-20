#pragma once

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv.hpp>

using namespace std;
using namespace cv;

class MotionVector{

public:
	MotionVector(void);
	MotionVector(double minDist, int maxCount = 100, double m_qualityLevel = 0.01);

public:
	void cal_motion_vector(const cv::Mat& src, std::vector<double>& motion_vector,  bool isPicture=false);

private:
	void motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
		vector<Point2f>& points_old, vector<Point2f>& points_new);

private:
	cv::Mat m_src;	//源图像
	std::vector<cv::Mat> m_buffer;	//图像缓存

	int m_last;		//缓存位置
	int m_maxCount;					// corners的最大个数

	double m_qualityLevel;			// 特征检测的等级
	double m_minDist;				// 两特征点之间的最小距离

	unsigned int m_motion_num;

	unsigned int m_buffer_length;
};