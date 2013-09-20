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

public:
	void cal_motion_vector(const cv::Mat& src, std::vector<double>& motion_vector,  bool isPicture=false);

private:
	void motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
		vector<Point2f>& points_old, vector<Point2f>& points_new);

private:
	cv::Mat m_src; //Դͼ��
	std::vector<cv::Mat> m_buffer; //ͼ�񻺴�

	unsigned int m_buffer_length; //���泤��
	int m_last; //��ʼ����λ��
	unsigned int m_point_record; //��һ��interest point��

	int m_maxCount; //corners��������
	double m_qualityLevel; //�������ĵȼ�
	double m_minDist; //��������֮�����С����

};