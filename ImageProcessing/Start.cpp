#include "Start.h"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

#define MAX_TRESHOLD 13000
#define ALLOWED_DIRECTION_DIFF 0.05
#define CLUSTER_COUNT 5
#define INF numeric_limits<double>::infinity()
#define INFINT numeric_limits<double>::infinity()
#define EPSILON 0.0000000000000000000001f
#define DISTANCE_BETWEEN_DIRECTION_VECTORS_THRESHOLD 10.0//d
int const comb[6][3] = { { 0,1,2 },{ 0,2,1 },{ 1,2,0 },{ 1,0,2 },{ 2,1,0 },{ 2,0,1 } };
double dirs[1][8] = { {0,45,90,135,180,225,270,315} };




int main(int argc, char** argv) {


	Mat page = imread("C:\\Users\\Asaf\\Desktop\\Gezer5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat word = imread("C:\\Users\\Asaf\\Desktop\\Capture.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	double inf = std::numeric_limits<double>::infinity();
	//int numberOfwantedClustersAtQuery = 5;

	pair<double,unordered_set<Cluster *> > p = buildClusters(word,DISTANCE_BETWEEN_DIRECTION_VECTORS_THRESHOLD);
	unordered_set<Cluster *> clusters2 = p.second;
	p= buildClusters(page, p.first);
	unordered_set<Cluster *> clusters1 = p.second;
	
	//vector<Cluster *> similars;
	//findSimilars(clusters1, clusters2.at(0), similars, MAX_TRESHOLD);

	
/*
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

	
	*/
	
	waitKey();
	



}

void calculateTrigularityList(unordered_set<Cluster *> clusters, Size imgSize ){
	

	Rect rect(0, 0, imgSize.width, imgSize.height);
	Subdiv2D subdiv(rect);
	
	for each (Cluster * p in clusters)
	{
		subdiv.insert(p->getClusterCenterPoint());
	}
	

	vector<Vec6f> trianglesList;
	subdiv.getTriangleList(trianglesList);
	Cluster *aC;
	Cluster *bC;
	Cluster *cC;
	
	for each (Vec6f t in trianglesList)
	{
		Point2f pts [3];
		bool findInClusters=true;
		pts[0] = Point2f(t[0], t[1]);
		pts[1] = Point2f(t[2], t[3]);
		pts[2] = Point2f(t[4], t[5]);
		
		for(int i=0; i<3 ;i++){
         if(pts[i].x>=imgSize.width ||pts[i].y >= imgSize.height || pts[i].x <= 0||pts[i].y <= 0)
		 {
            findInClusters=false;
			break;
		 }
      }

		if (findInClusters)
		{
		aC = findCluster(pts[0], clusters);
		bC = findCluster(pts[1], clusters);
		cC = findCluster(pts[2], clusters);
		
		aC->addNeighbour(bC);
		aC->addNeighbour(cC);
		bC->addNeighbour(aC);
		bC->addNeighbour(cC);
		cC->addNeighbour(aC);
		cC->addNeighbour(bC);
		}

	}
}


Cluster *findCluster(Point2f p, unordered_set<Cluster*> clusters) {
	for each (Cluster * c in clusters)
	{
		Point2f cP = c->getClusterCenterPoint();
		if (compareFloats(cP.x, p.x) && compareFloats(cP.y, p.y)) {
			return c;
		}
	}
	return NULL;
}




bool compareFloats(float a, float b) {
	return ((a - b < EPSILON )&& (b - a < EPSILON));
}






void findSimilars(unordered_set<Cluster *>, Cluster * c, unordered_set<Cluster *>& similars, double threshold) {

}


void findSimilars(vector<Cluster *> clusters, Cluster * c,vector<Cluster*>& similars, double threshold) {
	double min = INF;
	for each (Cluster * c1 in clusters) {
		double dis = 0;//getMinimalDistanceBetweenTriplets(c1, c);
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




pair<double, unordered_set<Cluster *>> buildClusters(Mat img,  double vectorDirectionThreshold,double distance) {
	cout << vectorDirectionThreshold << endl;
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	sft->detect(img, keypoints);
	sft->compute(img, keypoints, descriptors);
	unordered_set<Cluster *> initialClusters;
	unordered_set<Cluster *> joinedClusters;
	initClusters(keypoints, descriptors, initialClusters);
	calculateTrigularityList(initialClusters,img.size());
	double maxDis;
	maxDis=joinClustersFixedPoint(initialClusters,distance, vectorDirectionThreshold);
	return make_pair(maxDis, initialClusters);

}




double joinClustersFixedPoint(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold,double optionalDistance) {

	double maxDistance = -1;
	int previousCulstersSize = INFINT;

	maxDistance=max(maxDistance, joinClusters(clusters, vectorDirectionThreshold));

	while (clusters.size() < previousCulstersSize)
	{
		previousCulstersSize = clusters.size();
		maxDistance=max(maxDistance, joinClusters(clusters, vectorDirectionThreshold));
	}

	return maxDistance;

}


double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, double optionalDistance) {


	unordered_set<Cluster *> handled;
	unordered_set<Cluster *> toRemove;
	double maxDistance = -1;
	for each(Cluster * c in clusters) {
		unordered_set<Cluster *> neighbours = c->getNeighbours();
		handled.insert(c);
		for each(Cluster * n in neighbours) {
			if (handled.find(n) != handled.end() || !mergable(c, n, vectorDirectionThreshold, optionalDistance)) {
				continue;
			}
			double dist = euclideanDist(c->getClusterCenterPoint(), n->getClusterCenterPoint());
			maxDistance = max(dist, maxDistance);
			c->mergeClusters(n);
			toRemove.insert(n);

		}
	}
		for each (Cluster *c in toRemove)
		{
			clusters.erase(c);
		}

		return maxDistance;




	
}







pair<double,double> calculateDirection(Vec<uchar, 128> v) {
	
	Vec<double,8> dirSum;
	int entry;
	for (int i = 0; i <128; i++) {
		entry = i % 8;
		uchar val = v[i];
		dirSum[entry] += val;
	}
	int max = 0;
	for (int i = 1; i < 8; i++) {
		dirSum[i] = dirSum[i] / 16.0;
	} 
	Matx<double, 1, 8> m;
	transpose(dirSum, m);
	


	
	Vec<double, 8> anglesv;
	for (int i = 0; i < 8; i++) {
		anglesv[i] = 45 * i;
	}
	Matx<double, 1, 8> angles;
	transpose(anglesv, angles);
	Matx<double, 1, 8> xs;
	Matx<double, 1, 8> ys;
	polarToCart(m,angles,xs,ys,true);
	Matx<double, 1, 1> avgX;
	Matx<double, 1, 1> avgY;
	Matx<double, 1, 1> finalMag;
	Matx<double, 1, 1> finalAngle;
	reduce(xs,avgX,1,CV_REDUCE_AVG);
	reduce(ys,avgY,1,CV_REDUCE_AVG);
	cartToPolar(avgX,avgY,finalMag,finalAngle,true);
	double d = finalAngle(0, 0);
	double l = finalMag(0, 0);
	pair<double, double> p = make_pair(l,d);
	return p;

}




double euclideanDist(Point2f& p, Point2f& q) {
	Point diff = p - q;
	return cv::sqrt((double)(diff.x*diff.x) + (double)(diff.y*diff.y));
}

void initClusters(vector<KeyPoint> keypoints, Mat descriptors, unordered_set<Cluster*>& clusters) {
	for (unsigned int i = 0; i<keypoints.size(); i++) {
		Cluster *c = new Cluster();
		Vec<uchar, 128> v;
		Mat t = descriptors.row(i);
		c->addToClusterAndCalc(keypoints.at(i),t);
		clusters.insert(c);

	}

}

bool mergable(Cluster * a, Cluster * b, double vectorDirectionThreshold, double optionalDistance)
{
	if (optionalDistance > 0&& euclideanDist(a->getClusterCenterPoint(), b->getClusterCenterPoint())> optionalDistance) {
		return false;
	}
	//TODO: some calculation regarding the direction vectors
	pair<double,double> p1=calculateDirection(a->getClusterCenterDescriptor());
	pair<double, double> p2 = calculateDirection(b->getClusterCenterDescriptor());
	double angle = 180 - abs(abs(p1.second - p2.second) - 180);
	cout << angle << endl;
	return  abs(angle)< vectorDirectionThreshold;
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
