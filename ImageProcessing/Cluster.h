#pragma once
 #include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
using namespace std;
using namespace cv;
class Cluster
{
public:
	Cluster(void);
	~Cluster(void);

	Vec<uchar,128> getClusterCenterDescriptor();
	Point getClusterCenterPoint();
	Mat getSiftDescriptors();
	vector<KeyPoint> getClusterKeyPoints();
	void addToCluster(KeyPoint key,Vec<uchar,128> descriptor);
	void addToClusterAndCalc(KeyPoint key,Vec<uchar,128> descriptor);
	void calculateCenter();
	void addCluster(Cluster other);


private:
	vector<KeyPoint> keyPoints;
	Mat siftDescriptors;
	Vec<uchar,128> centerDescriptor; 
	Point centerPoint;
};

