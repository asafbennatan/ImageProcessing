
#include "Start.h"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

#define MAX_TRESHOLD 13000
#define INF numeric_limits<double>::infinity()
int const comb[6][3] = { { 0,1,2 },{ 0,2,1 },{ 1,2,0 },{ 1,0,2 },{ 2,1,0 },{ 2,0,1 } };

int main(int argc, char** argv) {


	Mat page = imread("C:\\Users\\Asaf\\Desktop\\Gezer5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat word = imread("C:\\Users\\Asaf\\Desktop\\Capture.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	double inf = std::numeric_limits<double>::infinity();
	int numberOfwantedClustersAtQuery = 3;

	pair<double,vector<Cluster *> > p = buildClusters(word, numberOfwantedClustersAtQuery);
	vector<Cluster *> clusters2 = p.second;
	vector<Cluster *> clusters1 = buildClusters(page, p.first);
	
	calculateNeighbours(clusters1);
	calculateNeighbours(clusters2);
	vector<Cluster *> similars;
	findSimilars(clusters1, clusters2.at(0), similars, MAX_TRESHOLD);

	

	Mat out2;
	vector<Point2f> k1;
	for each (Cluster * c in similars)
	{
		
			k1.push_back(c->getClusterCenterPoint());
			k1.push_back(c->getNeighbours().first->getClusterCenterPoint());
			k1.push_back(c->getNeighbours().second->getClusterCenterPoint());


	}
	vector<KeyPoint> converted;
	KeyPoint::convert(k1, converted);

	Mat out3;
	vector<Point2f> k2;
	for each (Cluster * c in clusters2)
	{

		k2.push_back(c->getClusterCenterPoint());
	}
	vector<KeyPoint> converted1;
	KeyPoint::convert(k2, converted1);

	drawKeypoints(page, converted, out2);
	imshow("page.jpg", out2);

	drawKeypoints(word, converted1, out3);
	imshow("word.jpg", out3);

	

	
	waitKey();
	



}


double getMinimalDistanceBetweenTriplets(Cluster * origin, Cluster * other) {
	vector<Cluster *> one;
	vector<Cluster * > two;
	one.push_back(origin);
	one.push_back(origin->getNeighbours().first);
	one.push_back(origin->getNeighbours().second);
	two.push_back(other);
	two.push_back(other->getNeighbours().first);
	two.push_back(other->getNeighbours().second);
	double min= numeric_limits<double>::infinity();
	for (int i = 0; i < 6; i++) {
		double sum = 0;
		for (int j = 0; j < 3; j++) {
			sum=sum+compareHist(one.at(j)->getClusterCenterDescriptor(), two.at(comb[i][j])->getClusterCenterDescriptor(), HISTCMP_CHISQR);
		}
		if (sum < min) {
			min = sum;
		}
		
	}
	return min;
}





void findSimilars(vector<Cluster *> clusters, Cluster * c,vector<Cluster*>& similars, double threshold) {
	double min = INF;
	for each (Cluster * c1 in clusters) {
		double dis = getMinimalDistanceBetweenTriplets(c1, c);
		if ( dis<= threshold) {
			similars.push_back(c1);
		}
		if (dis < min) {
			min = dis;
		}
	}
	cout << "min: " << min << endl;


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
	vector<Cluster *> initialClusters;
	vector<Cluster *> joinedClusters;
	initClusters(keypoints, descriptors,initialClusters);
	joinedClusters = joinClustersFixedPoint(initialClusters, distance);

	
	return joinedClusters;

}

pair<double,vector<Cluster *>> buildClusters(Mat img, int numberOfClusters) {
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	sft->detect(img, keypoints);
	sft->compute(img, keypoints, descriptors);
	vector<Cluster *> initialClusters;
	vector<Cluster *> joinedClusters;
	initClusters(keypoints, descriptors, initialClusters);
	double maxDis=joinClustersFixedPoint(initialClusters, numberOfClusters);
	return make_pair(maxDis, initialClusters);

}


vector<Cluster *> joinClustersFixedPoint(vector<Cluster *>& clusters, double distance) {

	vector<Cluster *> out = joinClusters(clusters, distance);
	if (out.size() == clusters.size()) {
		return out;
	}

	return joinClustersFixedPoint(out, distance);



}

double joinClustersFixedPoint(vector<Cluster *>& clusters, int numberOfClusters) {

	double maxDistance = -1;
	while (clusters.size() > numberOfClusters)
	{
		maxDistance=max(maxDistance, joinClusters(clusters));
	}

	return maxDistance;

}


double joinClusters(vector<Cluster *>& clusters) {
	Point2f ap;
	Point2f bp;
	double dist;
	double minDis = numeric_limits<double>::infinity();
	int minIndexJ = -1;
	int minIndexI = -1;
	bool notSameDirection = false;
	for (unsigned int i = 0; i < clusters.size(); i++) {
		
		Cluster *a = clusters.at(i);
		for (unsigned int j = i + 1; j < clusters.size(); j++) {
			
			Cluster *b = clusters.at(j);
			ap = a->getClusterCenterPoint();
			bp = b->getClusterCenterPoint();
			dist = euclideanDist(ap, bp);
			bool sameDir = sameDirection(a->getClusterCenterDescriptor(), b->getClusterCenterDescriptor());
			if (sameDir) {
				if (dist <= minDis||notSameDirection) {

					minDis = dist;
					minIndexJ = j;
					minIndexI = i;


				}
			}
			else {
				if (dist <= minDis&& (minIndexI==-1|| notSameDirection)) {

					minDis = dist;
					minIndexJ = j;
					minIndexI = i;
					notSameDirection = true;


				}
				
			}
			
		}
		
	}
	

	clusters.at(minIndexI)->mergeClusters(clusters.at(minIndexJ));
	clusters.erase(clusters.begin()+minIndexJ);
	return minDis;



	
}


vector<Cluster *> joinClusters(vector<Cluster *>& clusters, double distance) {
	unordered_set<int> joinedIndexs;
	vector<Cluster *> out;
	Point2f ap;
	Point2f bp;
	double dist;
	for (unsigned int i = 0; i < clusters.size(); i++) {
		if (joinedIndexs.find(i) != joinedIndexs.end()) {
			continue;
		}
		Cluster *a = clusters.at(i);
		for (unsigned int j = i + 1; j < clusters.size(); j++) {
			
			if (joinedIndexs.find(j) != joinedIndexs.end()) {
				continue;
			}
			Cluster *b = clusters.at(j);
			ap = a->getClusterCenterPoint();
			bp = b->getClusterCenterPoint();
			dist = euclideanDist(ap, bp);
			if (dist <= distance && sameDirection(a->getClusterCenterDescriptor(),b->getClusterCenterDescriptor())) {
				a->mergeClusters(b);
				joinedIndexs.insert(j);
				joinedIndexs.insert(i);
			}
		}
		out.push_back(a);
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


double euclideanDist(Point2f& p, Point2f& q) {
	Point diff = p - q;
	return cv::sqrt((double)(diff.x*diff.x) + (double)(diff.y*diff.y));
}

void initClusters(vector<KeyPoint> keypoints, Mat descriptors, vector<Cluster*>& clusters) {
	for (unsigned int i = 0; i<keypoints.size(); i++) {
		Cluster *c = new Cluster();
		Vec<uchar, 128> v;
		Mat t = descriptors.row(i);
		c->addToClusterAndCalc(keypoints.at(i),t);
		clusters.push_back(c);

	}

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
void calculateNeighbours(vector<Cluster *> clusters) {
	if (clusters.size() < 2) {
		return;
	}
	if (clusters.size() ==2) {
		Cluster * a = clusters.at(0);
		Cluster * b = clusters.at(1);
		a->setNeighbours(b, 0);
		b->setNeighbours(a, 0);
		return;
	}
	for(int i=0;i<clusters.size();i++)
	{
		vector<Cluster *> others(clusters);
		others.erase(others.begin()+ i);
		setNearestTwoNeighbours(others, clusters.at(i));
	}


}

void setNearestTwoNeighbours(vector<Cluster *> clusters, Cluster *c) {
	Cluster * min=0;
	Cluster * min2=0;
	
	double minDis= numeric_limits<double>::infinity();
	double minDis2 = numeric_limits<double>::infinity();
	for each (Cluster * cluster in clusters)
	{
		if (cluster != c) {
			double dis = euclideanDist(c->getClusterCenterPoint(), cluster->getClusterCenterPoint());
			if (dis < minDis) {
				min2 = min;
				min = cluster;
				minDis2 = minDis;
				minDis = dis;
			}
			else {
				if (dis < minDis2) {
					minDis2 = dis;
					min2 = cluster;
				}
			}
			
		}
		
	}
	
	c->setNeighbours(min, min2);
}






