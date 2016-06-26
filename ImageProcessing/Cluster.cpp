#include "Cluster.h"


Cluster::Cluster(void)
{
//	siftDescriptors.create(Size(128, 1), CV_16UC1);
}


Cluster::~Cluster(void)
{
}



Mat Cluster::getClusterCenterDescriptor(){
	return centerDescriptor;
}
Point2f Cluster::getClusterCenterPoint() {
	return centerPoint;
}

Mat Cluster::getSiftDescriptors(){
	return siftDescriptors;

}
vector<KeyPoint> Cluster::getClusterKeyPoints(){
	return keyPoints;
}

void Cluster::addToCluster(KeyPoint key,Mat descriptor){
	keyPoints.push_back(key);
	siftDescriptors.push_back(descriptor);
	//cout << "addToCluser" << endl;
	printClusterData();
}

void Cluster::calculateCenter(){
	if (siftDescriptors.rows == 1 ){
		centerDescriptor = siftDescriptors.row(0);
		centerPoint.x = keyPoints.front().pt.x;
		centerPoint.y = keyPoints.front().pt.y;
		
	}
	else
		{
	reduce(siftDescriptors, centerDescriptor, 0,CV_REDUCE_AVG, -1);
	float x=0;
	float y=0;
	for each (KeyPoint key in keyPoints)
	{
		x += key.pt.x;
		y += key.pt.y;
	}
	x = x / keyPoints.size();
	y = y / keyPoints.size();
	centerPoint.x = x;
	centerPoint.y = y;
		}
}

void Cluster::mergeClusters(Cluster * other)
{
	for (unsigned int i = 0; i < other->keyPoints.size(); i++) {
		addToCluster(other->keyPoints.at(i), other->siftDescriptors.row(i));
	}
	calculateCenter();
	//cout << "after mergeClusters" << endl;
	printClusterData();
}

void Cluster::setNeighbours(Cluster * n1, Cluster * n2)
{
	neighbours = make_pair(n1, n2);
}

pair<Cluster*, Cluster*> Cluster::getNeighbours()
{
	return neighbours;
}



void Cluster::addToClusterAndCalc(KeyPoint key,Mat descriptor){
	addToCluster(key,descriptor);
	calculateCenter();
	//cout << "addToClusterAndCalc" << endl;
	printClusterData();
}

void Cluster::printClusterData()
{
	//cout << "number of points: " << keyPoints.size() << endl;
	//cout << "center: "  << centerPoint.x << "  "  << centerPoint.y << endl;
}

