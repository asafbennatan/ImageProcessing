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


#include <unordered_set>
using namespace std;
using namespace cv;

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
vector<vector<Cluster*>> match(vector<Cluster*> page,vector<Cluster*> query,double treshold);
void findMinDistTriangle(pair<Point2f[3],Point2f[3]>& mtch, Point2f wordPts [3],vector<Vec6f>& TrianglesList,Size imgSize);
