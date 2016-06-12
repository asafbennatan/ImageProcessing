#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Cluster.h"
#include <unordered_set>

using namespace std;
using namespace cv;
vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<uchar, 128> calculateGrid(Mat cell);
double compare(vector<uchar> a, vector<uchar> b);

int main(int argc, char** argv) {


	Mat img_1 = imread("C:\\Users\\Ilana\\Desktop\\Rashi.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("C:\\Users\\Ilana\\Desktop\\smalll.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int gridW = 9;
	int gridH = 9;

	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints_1;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_1;
	Mat descriptors_2;


	sft->detect(img_1, keypoints_1);
	sft->detect(img_2, keypoints_2);
	sft->compute(img_1, keypoints_1, descriptors_1);
	sft->compute(img_2, keypoints_2, descriptors_2);

	Mat out0;
	drawKeypoints(img_1, keypoints_1/* temp*/, out0);
	imshow("KeyPoint0.jpg", out0);
	imwrite("KeyPoint0.jpg", out0);

	Mat out1;
	drawKeypoints(img_2, keypoints_2/* temp*/, out1);
	imshow("KeyPoint1.jpg", out1);
	imwrite("KeyPoint1.jpg", out1);



	//vector<KeyPoint> temp;
	//for (int i=0; i<1;i++)
	//	temp.push_back(keypoints_1.at(i));

	//for (int i=88; i<100;i++)
	//	temp.push_back(keypoints_1.at(i));
	//
	//cout << descriptors_1.rowRange(0,1) << endl;
	//cout << descriptors_1.rowRange(150,171) << endl;
	//
	waitKey();

	vector<uchar> id1 = getIdentificationVector(img_1, gridW, gridH);
	vector<uchar> id2 = getIdentificationVector(img_2, gridW, gridH);
	double res = compare(id1, id2);
	cout << "dis is " << res << endl;
	waitKey();



}


vector<Cluster> buildClusters(Mat img,double distance) {
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	sft->detect(img, keypoints);
	sft->compute(img, keypoints, descriptors);
	vector<Cluster> clusters = initClusters(keypoints, descriptors);
	clusters = joinClustersFixedPoint(clusters,distance);
	return clusters;

}


vector<Cluster> joinClustersFixedPoint(vector<Cluster> clusters,double distance) {

	vector<Cluster> out=joinClusters(clusters,distance);
	if (out.size() == clusters.size()) {
		return out;
	}

	joinClustersFixedPoint(out,distance);

}


vector<Cluster> joinClusters(vector<Cluster> clusters,double distance) {
	unordered_set<int> joinedIndexs;
	for (int i = 0; i < clusters.size(); i++) {
		if (joinedIndexs.find(i) == joinedIndexs.end()) {
			continue;
		}
		for (int j = i+1; j < clusters.size(); j++) {
			if (joinedIndexs.find(j) == joinedIndexs.end()) {
				continue;
			}
			Cluster a = clusters.at(i);
			Cluster b = clusters.at(j);
			Point ap = a.getClusterCenterPoint();
			Point bp = a.getClusterCenterPoint();
			if (euclideanDist(ap, bp) <= distance) {
				a.addCluster(b);
				joinedIndexs.insert(j);
			}
			
		}
	}
	

	
}


double euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

vector<Cluster> initClusters(vector<KeyPoint> keypoints, Mat descriptors) {
	vector<Cluster> clusters;
	for (int i = 0; i<keypoints.size(); i++) {
		Cluster c;
		c.addToClusterAndCalc(keypoints.at(i), descriptors.row(i));

	}
	return clusters;

}


vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight) {
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	int imageWidth = image.cols;
	int imageHeight = image.rows;

	int cellWidth = imageWidth / gridWidth;
	int cellHeight = imageHeight / gridHeight;

	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;

	vector<vector<Mat > >gridMultiplePoints(gridWidth, vector<Mat>(gridHeight));
	sft->detect(image, keypoints_1);
	sft->compute(image, keypoints_1, descriptors_1);



	for (int k = 0; k < keypoints_1.size(); k++) {
		KeyPoint key = keypoints_1.at(k);
		int i = floor(key.pt.x) / cellWidth;
		int j = floor(key.pt.y) / cellHeight;
		Point p(i, j);
		cout << "point " << i << "," << j << endl;

		Mat m = descriptors_1.rowRange(k, k + 1);

		gridMultiplePoints.at(i).at(j).push_back(m);




	}
	vector<uchar> grid;
	for (int i = 0; i < gridWidth; i++) {
		for (int j = 0; j < gridHeight; j++) {
			Mat m = gridMultiplePoints.at(i).at(j);
			if (m.empty()) {
				m = Mat::zeros(Size(128, 1), CV_32F);
			}
			Vec<uchar, 128> d = calculateGrid(m);
			grid.insert(end(grid), begin(d.val), end(d.val));

		}
	}
	grid.shrink_to_fit();
	return grid;





}

Vec<uchar, 128> calculateGrid(Mat cell) {
	if (cell.rows < 2) {
		return cell;
	}
	reduce(cell, cell, 0, CV_REDUCE_AVG, -1);
	return cell;

}

double compare(vector<uchar> a, vector<uchar> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		double v1 = a.at(i);
		double v2 = b.at(i);
		if (v1 != 0 && v2 != 0) {
			sum += (pow((v1 - v2), 2.0) / (v1 + v2));

		}
	}
	sum = sum / 2;
	sum = exp(sum);
	return sum;
}