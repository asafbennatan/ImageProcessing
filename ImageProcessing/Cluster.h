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

	Mat getClusterCenterDescriptor();
	Point getClusterCenterPoint();
	Mat getSiftDescriptors();
	vector<KeyPoint> getClusterKeyPoints();
	void addToCluster(KeyPoint key,Mat descriptor);
	void addToClusterAndCalc(KeyPoint key,Mat descriptor);
	void calculateCenter();
	void addCluster(Cluster *other);
	void addNeighbour(Cluster *other);


private:
	vector<KeyPoint> keyPoints;
	Mat siftDescriptors;
	Mat centerDescriptor; 
	Point centerPoint;
	vector<Cluster *> neighbours;
};

