#pragma once

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv.hpp>

using namespace std;
using namespace cv;

class MotionFlow{

public:
	MotionFlow(void);
	MotionFlow(int maxCount, double qualityLevel = 0.01, double minDist = 30);
	~MotionFlow(void);

public:
	void cal_motion_vector(const cv::Mat& src, std::vector<double>& mf_feature,  bool isPicture=false);

private:
	void motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
		vector<Point2f>& points_old, vector<Point2f>& points_new);
	void cal_feature();

private:
	cv::Mat m_src; //源图像
	std::vector<cv::Mat> m_buffer; //图像缓存

	unsigned int m_buffer_length; //缓存长度
	int m_last; //起始缓存位置
	unsigned int m_point_record; //上一个interest point数

	int m_maxCount; //corners的最大个数
	double m_qualityLevel; //特征检测的等级
	double m_minDist; //两特征点之间的最小距离

	std::vector<double> m_mf_feature; //运动光流特征
	std::vector<double> m_flow_vector; //光流向量

};