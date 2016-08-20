#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Cluster.h"
#include <algorithm> 
#include <set>
#include <iterator>

#include "boost/graph/vf2_sub_graph_iso.hpp"
#include "boost/filesystem.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <unordered_set>

#define DISTANCE_BETWEEN_DIRECTION_VECTORS_THRESHOLD 15.0//d
#define DISTANCE_BETWEEN_CLUSTERS_THRESHOLD 25000
using namespace std;
using namespace cv;
typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS> graph_type;


vector<double> getIdentificationVector(Mat image, int gridWidth, int gridHeight);
Vec<double, 128> calculateGrid(Mat cell);
double compare(Vec<double,128> a, Vec<double,128> b);
double euclideanDist(Point2f& p, Point2f& q);
pair<double, unordered_set<Cluster *>> buildClusters(Mat img,  double vectorDirectionThreshold,double distance =-1);
double joinClustersFixedPoint(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, Size imgSize,Mat img, double optionalDistance=-1);
void initClusters(vector<KeyPoint> keypoints, Mat descriptors, unordered_set<Cluster*>& clusters);
bool mergable(Cluster * a,Cluster *b,double vectorDirectionThreshold, double optionalDistance=-1);
pair<double,double> calculateDirection(Vec<double, 128> v);
void findSimilars(vector<Cluster *> clusters, Cluster * c,vector<Cluster*>& similars, double threshold);

void calcTriangularityList (vector<Vec6f>& trianglesList, unordered_set<Cluster *>& clusters, Size imgSize);
void setClusterNeighboursByTriangles (vector<Vec6f>& trianglesList, unordered_set<Cluster *>& clusters, Size imgSize);

Cluster *findCluster(Point2f p, unordered_set<Cluster*> clusters);
bool compareFloats(float a, float b);
vector<vector<int>> nChooseKPermutations(vector<int> arr, int k);
void nChooseKPermutationsRec(int offset, int k, vector<int> arr, vector<vector<int>> &combs, vector<int> &current);
double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, Size imgSize,Mat img, double optionalDistance = -1);
//double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, double optionalDistance = -1);
void findSimilars(unordered_set<Cluster *>, Cluster * c, unordered_set<Cluster *>& similars, double threshold);
double calculateDistanceBetweenKClusters(vector<Cluster *>page, vector<Cluster*> query);
double calculateMinDistanceBetweenKClusters(vector<Cluster *>page, vector<Cluster*> query);
vector<vector<Cluster*> > permGenerator(vector<Cluster *> clusters, int k);
vector<vector<Cluster*>> match(vector<Cluster*> page,vector<Cluster*> query,double treshold,double minDist);
double calculateClusterMaxDistance(vector<Cluster*> clusters);
double calculateClusterMaxNeighborDistance(unordered_set<Cluster * > clusters);
void findMinDistTriangle(pair<Point2f[3],Point2f[3]>& mtch, Point2f wordPts [3],vector<Vec6f>& TrianglesList,Size imgSize);
template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct pair_hash {
	std::size_t operator () (const std::pair<float, float> &p) const {
		size_t h = std::hash<float>{}(p.first);
		hash_combine(h, p.second);

		// Mainly for demonstration purposes, i.e. works but is overly simple
		// In the real world, use sth. like boost.hash_combine
		return h;
	}
};
struct pair_eq {
	bool operator()(const pair<float, float> t1,const  pair<float, float> t2) const
	{

		return compareFloats(t1.first, t2.first) && compareFloats(t1.second, t2.second);
	}
};


void getSubIsomorphicGraphs(vector<Vec6f>& queryT, vector<Vec6f>& pageT, int queryVertexSize, int pageVertexSize);
void test(graph_type a, graph_type b);
graph_type createGraph(vector<Vec6f>& tList, int numberOfvertex);
int getVertexNumber(boost::unordered_map<pair<float, float>, int, pair_hash, pair_eq> & map, float x, float y, int & counter);
Mat runAlgorithem(string pagePath, string queryPath, double distance_between_direction_vectors_thresold = DISTANCE_BETWEEN_DIRECTION_VECTORS_THRESHOLD, double distance_between_clusters_threshold = DISTANCE_BETWEEN_CLUSTERS_THRESHOLD);
Cluster * findBestCluster(Cluster * c, vector<Cluster*> clusters, unordered_set<int> ignore, double clustersDistanceTreshold, double maxQueryDist);
