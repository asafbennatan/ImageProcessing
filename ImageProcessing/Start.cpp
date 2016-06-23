#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Cluster.h"
#include "Start.h"
#include <unordered_set>

using namespace std;
using namespace cv;


int main(int argc, char** argv) {


	Mat page = imread("C:\\Users\\Asaf\\Desktop\\Gezer5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat word = imread("C:\\Users\\Asaf\\Desktop\\Gezer5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int gridW = 9;
	int gridH = 9;

	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints_1;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_1;
	Mat descriptors_2;


	sft->detect(page, keypoints_1);
	sft->detect(word, keypoints_2);
	sft->compute(page, keypoints_1, descriptors_1);
	sft->compute(word, keypoints_2, descriptors_2);

	Mat out0;
	drawKeypoints(page, keypoints_1/* temp*/, out0);
	imshow("KeyPoint0.jpg", out0);
	imwrite("KeyPoint0.jpg", out0);

	Mat out1;
	drawKeypoints(word, keypoints_2/* temp*/, out1);
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
	//waitKey();
	double inf = std::numeric_limits<double>::infinity();

	double distance = inf;
	double threshold = inf;
	vector<Cluster *> clusters1=buildClusters(page, distance);
	vector<Cluster *> clusters2 = buildClusters(word, distance);
	vector<Cluster *> found;
	for each (Cluster * cluster in clusters2)
	{
		vector<Cluster *> similars = findSimilars(clusters1,cluster, threshold);
		if (similars.size() > 0) {
			found = similars;
			break;
		}


	}
	Mat out2;
	for each (Cluster * c in found)
	{
		
		drawKeypoints(page,c->getClusterKeyPoints(), out2);
		
		

	}
	imwrite("KeyPoint2.jpg", out2);
	imshow("KeyPoint2.jpg", out2);
	



}

vector<Cluster * > findSimilars(vector<Cluster *> clusters, Cluster * c,double threshold) {
	vector<Cluster *> similars;
	for each (Cluster * c1 in clusters) {
		if (compare(c1->getClusterCenterDescriptor(), c->getClusterCenterDescriptor()) < threshold) {
			similars.push_back(c1);
		}
	}
	return similars;
}

bool match(Cluster * a, Cluster * b) {
	return true;
}


vector<Cluster *> buildClusters(Mat img, double distance) {
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	sft->detect(img, keypoints);
	sft->compute(img, keypoints, descriptors);
	vector<Cluster *> clusters = initClusters(keypoints, descriptors);
	clusters = joinClustersFixedPoint(clusters, distance);
	return clusters;

}


vector<Cluster *> joinClustersFixedPoint(vector<Cluster *> clusters, double distance) {

	vector<Cluster *> out = joinClusters(clusters, distance);
	if (out.size() == clusters.size()) {
		return out;
	}

	joinClustersFixedPoint(out, distance);

}


vector<Cluster *> joinClusters(vector<Cluster *> clusters, double distance) {
	unordered_set<int> joinedIndexs;
	vector<Cluster *> out;
	for (unsigned int i = 0; i < clusters.size(); i++) {
		if (joinedIndexs.find(i) != joinedIndexs.end()) {
			continue;
		}
		for (int j = i + 1; j < clusters.size(); j++) {
			if (joinedIndexs.find(j) != joinedIndexs.end()) {
				continue;
			}
			Cluster *a = clusters.at(i);
			Cluster *b = clusters.at(j);
			Point ap = a->getClusterCenterPoint();
			Point bp = a->getClusterCenterPoint();
			if (euclideanDist(ap, bp) <= distance&& sameDirection(a->getClusterCenterDescriptor(),b->getClusterCenterDescriptor())) {
				a->addCluster(b);
				out.push_back(a);
				joinedIndexs.insert(j);
			}

		}
	

	}
	return out;
}

bool sameDirection(Vec<uchar, 128> v1, Vec<uchar, 128> v2) {
	int i = calculateDirection(v1);
	int j = calculateDirection(v2);
	return i == j;
}


int calculateDirection(Vec<uchar, 128> v) {
	uchar dirSum[8];
	for (int i = 0; i <128; i++) {
		int entry = i % 8;
		dirSum[entry] += v[i];
	}
	int max = 0;
	for (int i = 1; i < 8; i++) {
		if (dirSum[i] > dirSum[max]) {
			max = i;
		}
	}
	return max;
}


double euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt((double)(diff.x*diff.x) + (double)(diff.y*diff.y));
}

vector<Cluster *> initClusters(vector<KeyPoint> keypoints, Mat descriptors) {
	vector<Cluster *> clusters;
	for (unsigned int i = 0; i<keypoints.size(); i++) {
		Cluster *c = new Cluster();
		Vec<uchar, 128> v;
		Mat t = descriptors.row(i);
		//transpose(t, v);
		c->addToClusterAndCalc(keypoints.at(i),t);
		clusters.push_back(c);

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



	for (unsigned int k = 0; k < keypoints_1.size(); k++) {
		KeyPoint key = keypoints_1.at(k);
		int i = (int)(floor(key.pt.x) / cellWidth);
		int j = (int)(floor(key.pt.y) / cellHeight);
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

double compare(Vec<uchar,128> a, Vec<uchar,128> b) {
	double sum = 0;
	for (unsigned int i = 0; i < 128; i++) {
		double v1 = a[i];
		double v2 = b[i];
		if (v1 != 0 && v2 != 0) {
			sum += (pow((v1 - v2), 2.0) / (v1 + v2));

		}
	}
	sum = sum / 2;
	sum = exp(sum);
	return sum;
}