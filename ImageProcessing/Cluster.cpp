#include "Cluster.h"


Cluster::Cluster(void)
{
}


Cluster::~Cluster(void)
{
}



Vec<uchar,128> Cluster::getClusterCenterDescriptor(){
	return centerDescriptor;
}
Point Cluster::getClusterCenterPoint() {
	return centerPoint;
}

Mat Cluster::getSiftDescriptors(){
	return siftDescriptors;

}
vector<KeyPoint> Cluster::getClusterKeyPoints(){
	return keyPoints;
}

void Cluster::addToCluster(KeyPoint key,Vec<uchar,128> descriptor){
	keyPoints.push_back(key);
	siftDescriptors.push_back(descriptor);

}

void Cluster::calculateCenter(){
		if (siftDescriptors.rows < 2) {
		centerDescriptor= siftDescriptors;
		centerPoint = keyPoints.front;
		return;
	}
	reduce(siftDescriptors, centerDescriptor, 0,CV_REDUCE_AVG, -1);
	float x=0;
	float y=0;
	for each (KeyPoint key in keyPoints)
	{
		x += key.pt.x;
		y += key.pt.y;
	}
	x = x / keyPoints.size;
	y = y / keyPoints.size;
	centerPoint.x = x;
	centerPoint.y = y;


}

void Cluster::addCluster(Cluster other)
{
	for (int i = 0; i < other.getClusterKeyPoints().size();i++) {
		addToCluster(other.getClusterKeyPoints().at(i), other.getSiftDescriptors().at(i));
	}
	calculateCenter();
}

void Cluster::addToClusterAndCalc(KeyPoint key,Vec<uchar,128> descriptor){
	addToCluster(key,descriptor);
	calculateCenter();
}
