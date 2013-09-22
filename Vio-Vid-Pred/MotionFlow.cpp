#include "stdafx.h"

#include "MotionFlow.h"

using namespace std;
using namespace cv;

MotionFlow::MotionFlow(void){

	/*������*/
	const unsigned int maxCount = 16;
	const double qualityLevel = 0.01;
	const double minDist = 30;

	/*�������*/
	const unsigned int buffer_length = 2;
	const unsigned int last = 0;
	const unsigned int point_record = maxCount;

	m_buffer_length = buffer_length;
	m_point_record = point_record;
	m_last = last;

	m_maxCount = maxCount;
	m_qualityLevel = qualityLevel;
	m_minDist = minDist;

	m_buffer.clear();
	m_mf_feature.clear();
	m_flow_vector.clear();

}

MotionFlow::MotionFlow(int maxCount, double qualityLevel, double minDist){

	const unsigned int buffer_length = 2;
	const unsigned int last = 0;
	const unsigned int point_record = maxCount;

	m_buffer_length = buffer_length;
	m_point_record = point_record;
	m_last = last;

	m_maxCount = maxCount;
	m_qualityLevel = qualityLevel;
	m_minDist = minDist;

	m_buffer.clear();

}

MotionFlow::~MotionFlow(void){

	m_src.release();

	m_buffer.clear();

}

void  MotionFlow::cal_motion_vector(const cv::Mat& src, std::vector<double>& mf_feature, bool isPicture){

	/*���ͼ��*/
	if(src.empty()){
		std::cerr << "Sorry, the source image is empty! " << std::endl;
		return;
	}
	src.copyTo(m_src);

	/*��ʼ��*/
	if(m_buffer.empty()){

		/*��ʼ������*/
		for(unsigned int i=0; i<m_buffer_length; i++){
			m_buffer.push_back(cv::Mat::zeros(src.rows, src.cols, CV_32FC1));
		}

		/*��ʼ����һ��ͼ��*/
		cv::cvtColor(m_src, m_buffer[m_last], CV_BGR2GRAY);

		return;
	}

	/*���㻺���С*/
	unsigned int buffer_length = m_buffer.size();
	if(buffer_length < 2){
		std::cerr << "Sorry, the buffer length should be more than 2! " << std::endl;
		return;
	}

	int idx1 = m_last;	//ǰ֡����
	int idx2 = (m_last + 1) % buffer_length; //��֡����
	m_last = idx2;

	/*ת����ǰͼ��Ϊ�Ҷ�ͼ��*/
	cv::cvtColor(m_src, m_buffer[idx2], CV_BGR2GRAY);

	vector<Point2f> points_old; //ǰ֡�ؼ���
	vector<Point2f> points_new; //��֡�ؼ���

	/*�������Ȥ��*/
	motion_tracking(m_buffer[idx1], m_buffer[idx2], points_old, points_new);

	/*��ͼ*/
	if(isPicture){
		for (size_t i=0; i < points_new.size(); i++){
			cv::line(m_src, points_old[i], points_new[i], cv::Scalar(0, 0, 255), 2);
			cv::circle(m_src, points_new[i], 2, Scalar(255, 0, 255), -1);
			cv::circle(m_src, points_old[i], 2, Scalar(0, 255, 255), -1);
		}
		cv::imshow( "Motion Vector", m_src);
		if( cv::waitKey(1) >= 0 )
			return;
	}

	/*�����˶�����ֵ*/
	double temp_x(0.0);
	double temp_y(0.0);
	double temp_result(0.0);

	for(unsigned int i=0; i<points_new.size(); i++){
		temp_x = points_new[i].x-points_old[i].x;
		temp_y = points_new[i].y-points_old[i].y;
		temp_result = pow(temp_x, 2.0)+pow(temp_y, 2.0);
		m_flow_vector.push_back(temp_result);
	}

	cal_feature();
	mf_feature = m_mf_feature;

	m_flow_vector.clear();
	m_mf_feature.clear();
	points_old.clear();
	points_new.clear();
}

void MotionFlow::motion_tracking(cv::Mat& pre_frame, cv::Mat& next_frame,
	vector<Point2f>& points_old, vector<Point2f>& points_new){

		/*ͳһͼ���С*/
		cv::Size frame_size = pre_frame.size();
		cv::resize(next_frame, next_frame, frame_size, 0.0, 0.0, INTER_CUBIC);

		points_old.clear();
		points_new.clear();

		cv::Mat t_pre_frame;
		cv::Mat t_next_frame;

		/*�Ҷ�ת��*/
		if(pre_frame.channels() != 1){
			cv::cvtColor(pre_frame, t_pre_frame, CV_BGR2GRAY);
		}else{
			t_pre_frame = pre_frame;
		}
		if(next_frame.channels() != 1){
			cv::cvtColor(next_frame, t_next_frame, CV_BGR2GRAY);
		}else{
			t_next_frame = next_frame;
		}

		std::vector<cv::Point2f> points_old_temp;
		std::vector<cv::Point2f> points_new_temp;

		/*����ǰ֡����Ȥ��*/
		cv::goodFeaturesToTrack(t_pre_frame, points_old_temp, m_maxCount, m_qualityLevel, m_minDist);

		/*��ȷ����*/
		cv::Size winSize(5, 5);
		cv::Size zeroZone(-1,-1);
		TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
		cv::cornerSubPix(t_pre_frame, points_old_temp, winSize, zeroZone, criteria);

		/*������������֡����Ȥ��*/
		vector<uchar> status;
		vector<float> err;
		cv::calcOpticalFlowPyrLK(t_pre_frame, t_next_frame, points_old_temp, points_new_temp, status, err);

		/*���Ը���ʧ�ܵĵ�*/
		for(unsigned i=0; i<err.size(); i++){
			if(err[i] > 1.0 && status[i]){

				/*���Ծ�ͷ�л�*/
				if(static_cast<unsigned int>(abs(static_cast<double>(
					m_point_record - points_new_temp.size()))) < m_point_record/4){
						points_new.push_back(points_new_temp[i]);
						points_old.push_back(points_old_temp[i]);
				}
				m_point_record = points_new_temp.size();
			}
		}

}

void MotionFlow::cal_feature(){

	cv::Mat temp_mat = cv::Mat::zeros(1, m_flow_vector.size(), CV_64FC1);
	cv::Mat mean = cv::Mat::zeros(1, 1, CV_64FC1);
	cv::Mat stddev = cv::Mat::zeros(1, 1, CV_64FC1);
	double sum_fv(0.0);

	unsigned int temp(0);
	for(unsigned int i=0; i<m_mf_feature.size(); i++){
		temp_mat.at<double>(0,i) = m_flow_vector[i];
	}
	cv::meanStdDev(temp_mat, mean, stddev);
	sum_fv = cv::sum(temp_mat)[0];

	/*����������*/
	m_mf_feature.push_back(sum_fv);

	/*���������;�ֵ*/
	m_mf_feature.push_back(mean.at<double>(0,0));

	/*���������ͱ�׼��*/
	m_mf_feature.push_back(stddev.at<double>(0,0));

}