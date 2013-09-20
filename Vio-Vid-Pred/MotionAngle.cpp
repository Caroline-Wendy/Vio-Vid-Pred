#include "stdafx.h"

#include "MotionAngle.h"

using namespace std;
using namespace cv;

MotionAngle::MotionAngle(void){

	const unsigned int diff_threshold = 30;
	const unsigned int buffer_length = 4;
	const double mhi_duration = 1.0;

	m_src.release();
	m_dst.release();
	m_mhi.release();
	m_orient.release();
	m_mask.release();
	m_segmask.release();

	m_buffer.clear();
	m_storage.clear();

	/*mhiͼ��ĳ���ʱ��*/
	m_mhi_duration = mhi_duration;

	/*ȡֵ0~255, ������ֵ����Ϊ1*/
	m_diff_threshold = diff_threshold;

	/*�������ĳ��ȣ�����2*/
	m_buffer_length = buffer_length;

	/*�洢λ��*/
	m_last = 0;
}

MotionAngle::MotionAngle(const double mhi_duration, 
	const unsigned int diff_threshold, const unsigned int buffer_length){

	m_mhi.release();
	m_orient.release();
	m_mask.release();
	m_segmask.release();

	m_mhi_duration = mhi_duration;

	/*ȡֵ0~255, ������ֵ����Ϊ1*/
	m_diff_threshold = diff_threshold;
	m_buffer_length = buffer_length;

	m_last = 0;
}

MotionAngle::~MotionAngle(void){
	m_src.release();
	m_dst.release();
	m_mhi.release();
	m_orient.release();
	m_mask.release();
	m_segmask.release();

	m_buffer.clear();
	m_storage.clear();
}

void  MotionAngle::update_mhi(const cv::Mat& src, std::vector<double>& angle_vector, bool isPicture){

	/*�ж�ͼ��*/
	if(src.empty()){
		std::cerr << "Sorry, the source image is empty! " << std::endl;
		return;
	}
	src.copyTo(m_src);

	angle_vector.clear();

	double timestamp; //ʱ���(��)
	timestamp = (double)clock()/CLOCKS_PER_SEC;

	cv::Size size = m_src.size(); //Դͼ���С

	/*��ʼ��*/
	if(m_mhi.empty() || m_mhi.cols != size.width || m_mhi.rows != size.height ) {

		/*��ʼ������*/
		for(unsigned int i=0; i<m_buffer_length; i++){
			m_buffer.push_back(cv::Mat::zeros(src.rows, src.cols, CV_8UC1));
		}

		/*��ʼ��ͼ��*/
		m_mhi.release();
		m_orient.release();
		m_segmask.release();
		m_mask.release();

		m_mhi = cv::Mat::zeros( size.height, size.width, CV_32FC1); //�˶���ʷͼ��
		m_orient = cv::Mat::zeros( size.height, size.width, CV_32FC1); //�˶��ݶ�ͼ��
		m_segmask = cv::Mat::zeros( size.height, size.width, CV_32FC1);
		m_mask = cv::Mat::zeros(size.height, size.width, CV_8UC1);
	}

	unsigned int buffer_length = m_buffer.size();	//���泤��
	if(buffer_length < 2){
		std::cerr << "Sorry, the buffer length should be more than 2! " << std::endl;
		return;
	}

	int idx1 = m_last; //֡����
	int idx2 = (m_last + 1) % buffer_length; //��һ֡����

	/*ת����ǰͼ��Ϊ�Ҷ�ͼ��*/
	cv::cvtColor(m_src, m_buffer[m_last], CV_BGR2GRAY);

	m_last = idx2;
	cv::Mat silh; //�洢����ͼ��(silhouette)
	silh = m_buffer[idx2];

	/*��ȡ��֮֡�������*/
	cv::absdiff(m_buffer[idx1], m_buffer[idx2], silh);

	/*��ֵ��ͼ��*/
	cv::threshold(silh, silh, m_diff_threshold, 1, CV_THRESH_BINARY );

	/*�����˶���ʷͼ��*/
	cv::updateMotionHistory(silh, m_mhi, timestamp, m_mhi_duration);

	/*ת��MHIΪ8Uͼ��*/
	cv::convertScaleAbs(m_mhi, m_mask, 255./m_mhi_duration, 
		(m_mhi_duration - timestamp)*255./m_mhi_duration);

	/*�����˶��ݶȷ���(orient) & ��ʵ���������(mask)*/
	cv::calcMotionGradient(m_mhi, m_mask, m_orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	/*�ָ��˶�: �õ��˶����������; segmask���˶����������*/
	cv::segmentMotion(m_mhi, m_segmask, m_storage, timestamp, MAX_TIME_DELTA );

	cv::Rect comp_rect;	//����λ��
	bool isWhole(true);	//�ж�����
	double angle;			//�Ƕ�

	for(int i = -1; i < static_cast<int>(m_storage.size()); i++){
		if(i<0) {
			comp_rect = cv::Rect( 0, 0, size.width, size.height );
			isWhole = true;
		}else {
			comp_rect = m_storage[i];
			if( comp_rect.width + comp_rect.height < 100 ) //����С������
				continue;
			isWhole = false;
		}

		/*ѡȡROI����*/
		cv::Mat mhi_roi = m_mhi(comp_rect);
		cv::Mat orient_roi = m_orient(comp_rect);
		cv::Mat mask_roi = m_mask(comp_rect);

		/*���㷽��*/
		angle = cv::calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, m_mhi_duration);
		angle = 360.0 - angle;  //���Ͻ�ԭ��

		if(isPicture){
			draw_picture(m_src, comp_rect, silh, isWhole, angle);		//��ͼ
		}

		angle /= 360;
		angle_vector.push_back(angle);
	}

	if(isPicture){
		cv::imshow( "Motion Angle", m_src);
		if( cv::waitKey(1) >= 0 )
			return;
	}
}

void MotionAngle::draw_picture(cv::Mat& dst, const cv::Rect comp_rect, const cv::Mat& silh, bool isWhole, double angle){

	double magnitude;
	double count;
	cv::Point center;
	cv::Scalar color;


	if(isWhole) {	//����
		color = cv::Scalar(255,0,255);
		magnitude = 100;
	}else{	//�ֲ�
		color = CV_RGB(255,255,0);
		magnitude = 30;
	}

	/*����ROI����*/
	cv::Mat silh_roi = silh(comp_rect);

	/*������������*/
	count = cv::norm(silh_roi, NORM_L1); 


	// check for the case of little motion
	if( count < comp_rect.width*comp_rect.height*0.05 )
		return;

	// draw a clock with arrow indicating the direction
	center = cv::Point( (comp_rect.x + comp_rect.width/2), (comp_rect.y + comp_rect.height/2) );
	cv::Point edge_point = cv::Point(cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
		cvRound( center.y - magnitude*sin(angle*CV_PI/180)));

	cv::circle( dst, center, static_cast<int>(magnitude*1.2), color, 2, CV_AA, 0 );
	cv::line( dst, center, edge_point, color, 2, CV_AA, 0 );
}