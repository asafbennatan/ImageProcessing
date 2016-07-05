#pragma once
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>
#include <unordered_set>


using namespace std;
using namespace cv;
class Cluster
{
public:
	Cluster(void);
	~Cluster(void);

	Mat getClusterCenterDescriptor();
	Point2f getClusterCenterPoint();
	Mat getSiftDescriptors();
	vector<KeyPoint> getClusterKeyPoints();
	void addToCluster(KeyPoint key,Mat descriptor);
	void addToClusterAndCalc(KeyPoint key,Mat descriptor);
	void calculateCenter();
	void mergeClusters(Cluster *other);
	void addNeighbour(Cluster *n1);
	unordered_set<Cluster *> getNeighbours();
	void printClusterData();



private:
	vector<KeyPoint> keyPoints;
	Mat siftDescriptors;
	Mat centerDescriptor; 
	Point2f centerPoint;
	unordered_set<Cluster *> neighbours;
	
};