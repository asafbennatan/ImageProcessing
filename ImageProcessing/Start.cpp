#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;
vector<uchar> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<uchar, 128> calculateGrid(Mat cell);
double compare(vector<uchar> a, vector<uchar> b);

int main(int argc, char** argv) {
	

	Mat img_1 = imread("C:\\Users\\Asaf\\Desktop\\img1.jpg", 1);
	Mat img_2 = imread("C:\\Users\\Asaf\\Desktop\\img2.jpg", 1);
	int gridW = 9;
	int gridH = 9;





	vector<uchar> id1 = getIdentificationVector(img_1, gridW, gridH);
	vector<uchar> id2 = getIdentificationVector(img_2, gridW, gridH);
	double res=compare(id1, id2);
	cout << "dis is " << res << endl;
	waitKey();
	


}


vector<uchar> getIdentificationVector(Mat image,int gridWidth,int gridHeight) {
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	int imageWidth = image.cols;
	int imageHeight = image.rows;

	int cellWidth = imageWidth / gridWidth;
	int cellHeight = imageHeight / gridHeight;

	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	vector<vector<Mat > >gridMultiplePoints(gridWidth,vector<Mat>(gridHeight));
	sft->detect(image, keypoints_1);
	sft->compute(image, keypoints_1, descriptors_1);


	for (int k = 0; k < keypoints_1.size();k++) {
		KeyPoint key = keypoints_1.at(k);
		int i = floor(key.pt.x) / cellWidth;
		int j = floor(key.pt.y) / cellHeight;
		Point p(i, j);
		cout << "point " << i << "," << j << endl;
		
		Mat m=descriptors_1.rowRange(k, k + 1);
		
		gridMultiplePoints.at(i).at(j).push_back(m);

		
		

	}
	vector<uchar> grid;
	for (int i=0; i < gridWidth; i++) {
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

Vec<uchar,128> calculateGrid(Mat cell) {
	if (cell.rows < 2) {
		return cell;
	}
	reduce(cell, cell, 0,CV_REDUCE_AVG, -1);
	return cell;
	
}

double compare(vector<uchar> a, vector<uchar> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		uchar v1 = a.at(i);
		uchar v2 = b.at(i);
		if (v1 != 0 && v2 != 0) {
			sum += (pow((v1 - v2), 2) / (v1 + v2));

		}
	}
	sum = sum / 2;
	sum = exp(sum);
	return sum;
}