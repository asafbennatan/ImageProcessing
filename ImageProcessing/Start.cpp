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

int const comb[6][3] = { { 0,1,2 },{ 0,2,1 },{ 1,2,0 },{ 1,0,2 },{ 2,1,0 },{ 2,0,1 } };
double dirs[1][8] = { {0,45,90,135,180,225,270,315} };
namespace fs = boost::filesystem;
int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "usage: app <pathToFolder>" << endl;
		return 1;
	}
	
	fs::path baseDir(argv[1]);
	string out = argv[2];
	fs::directory_iterator end_iter;
	vector<fs::path> pages;
	vector<fs::path> queries;
	string pageStarts("page");
	string queryStarts("query");
	string suffix(".jpg");

	if (fs::exists(baseDir) && fs::is_directory(baseDir))
	{
		for (fs::directory_iterator dir_iter(baseDir); dir_iter != end_iter; ++dir_iter)
		{
			if (fs::is_regular_file(dir_iter->status()))
			{
				fs::path p = dir_iter->path();
				string name = p.filename().string();
				if (boost::algorithm::starts_with(name, queryStarts) && boost::algorithm::ends_with(name,suffix)) {
					queries.push_back(p);

				}

				if (boost::algorithm::starts_with(name, pageStarts)&& boost::algorithm::ends_with(name, suffix)) {
					pages.push_back(p);

				}
			}
		}
		
		for each (fs::path pagePath in pages)
		{
			for each(fs::path queryPath in queries) {
			
				string outPath = out +pagePath.stem().string() + "_" + queryPath.stem().string() + ".jpg";
				Mat out=runAlgorithem(pagePath.string(), queryPath.string());
				cv::imwrite(outPath, out);
			}
		}
	}
	else {
		cout << "invalid path" << endl;
	}


}


Mat runAlgorithem(string pagePath, string queryPath,double distance_between_direction_vectors_thresold , double distance_between_clusters_threshold) {
	static const int arr[] = { 1,2,3,4,5,6,7,8,9 };
	vector<int> vec(arr, arr + sizeof(arr) / sizeof(arr[0]));




	Mat page = imread(pagePath, CV_LOAD_IMAGE_GRAYSCALE);
	Mat word = imread(queryPath, CV_LOAD_IMAGE_GRAYSCALE);
	double inf = std::numeric_limits<double>::infinity();

	pair<double, unordered_set<Cluster *> > p = buildClusters(word, distance_between_direction_vectors_thresold);
	unordered_set<Cluster *> wordClusters = p.second;
	p = buildClusters(page, distance_between_direction_vectors_thresold,p.first);
	unordered_set<Cluster *> pageClusters = p.second;
	

	vector<Vec6f> wordTrianglesList;
	calcTriangularityList(wordTrianglesList, wordClusters, word.size());
	vector<Vec6f> pageTrianglesList;
	calcTriangularityList(pageTrianglesList, pageClusters, page.size());
	vector<Cluster *> pageClustersVector;
	std::copy(pageClusters.begin(), pageClusters.end(), std::back_inserter(pageClustersVector));
	vector<Cluster *> queryClustersVector;
	std::copy(wordClusters.begin(), wordClusters.end(), std::back_inserter(queryClustersVector));
	getSubIsomorphicGraphs(wordTrianglesList, pageTrianglesList, queryClustersVector.size(), pageClustersVector.size());
	double maxQueryDis = calculateClusterMaxDistance(queryClustersVector);
	vector<vector<Cluster*>> matched = match(pageClustersVector, queryClustersVector, distance_between_clusters_threshold, maxQueryDis);
	cout << "number of returned groups: " << matched.size() << endl;
	for each (vector<Cluster*> clusters in matched)
	{

		Point2d max(-INFINT, -INFINT);
		Point2d min(INF, INF);
		vector<KeyPoint> points;

		for each (Cluster* c in clusters)
		{


			for each (KeyPoint point in c->getClusterKeyPoints())
			{
				points.push_back(point);
				//	if (point.pt.x > max.x) {
				///		max.x = point.pt.x;
				//	}
				//	if (point.pt.y > max.y) {
				//		max.y = point.pt.y;
				//	}
				//	if (point.pt.x< min.x) {
				//		min.x = point.pt.x;
				//	}
				//	if (point.pt.y < min.y) {
				//		min.y = point.pt.y;
				//	}
			}

		}
		/*cv::rectangle(
			page,
			min,max,
			cv::Scalar(0, 255, 255)
		);*/

		Mat out;
		drawKeypoints(page, points,out, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		return out;




	}
}



void getSubIsomorphicGraphs(vector<Vec6f>& queryT, vector<Vec6f>& pageT,int queryVertexSize,int pageVertexSize) {
	graph_type queryG=createGraph(queryT, queryVertexSize);
	graph_type pageG = createGraph(pageT, pageVertexSize);
	boost::vf2_print_callback<graph_type, graph_type> callback(queryG, pageG);
	boost::vf2_subgraph_iso(queryG, pageG, callback);
	
}

//void test(graph_type a, graph_type b) {
//	BGL_FORALL_VERTICES_T(v, a, graph_type) {
//		int index=boost::get(boost::vertex_index_t(), a, v);
//	}
//	
//
//}



graph_type createGraph(vector<Vec6f>& tList,int numberOfvertex) {
	graph_type graph(numberOfvertex);
	boost::unordered_map<pair<float,float>, int,pair_hash, pair_eq> map;

	int counter = 0;
	for each (Vec6f t in tList)
	{
		int one=getVertexNumber(map, t[0], t[1], counter);
		int two = getVertexNumber(map, t[2], t[3], counter);
		int three = getVertexNumber(map, t[4], t[5], counter);

		boost::add_edge(one,two,graph);
		boost::add_edge(one, three, graph);
		boost::add_edge(two, three, graph);
	}
	return graph;
}



int getVertexNumber(boost::unordered_map<pair<float, float>, int,pair_hash, pair_eq> & map ,float x,float y, int & counter) {
	pair<float, float> p = make_pair(x, y);
	boost::unordered_map<pair<float, float>, int,pair_hash, pair_eq>::const_iterator looked = map.find(p);
	int val = -1;
	if (looked == map.end()) {
		val = counter;
		map.insert(make_pair(p, val));
		counter++;
	}
	else {
		val = looked->second;
	}
	
return	val;
}

void calcTriangularityList (vector<Vec6f>& trianglesList, unordered_set<Cluster *>& clusters, Size imgSize)
{
	Rect rect(0, 0, imgSize.width, imgSize.height);
	Subdiv2D subdiv(rect);

	for each (Cluster * p in clusters)
	{
		subdiv.insert(p->getClusterCenterPoint());
	}
	
	subdiv.getTriangleList(trianglesList);
	
		
	
}

void setClusterNeighboursByTriangles (vector<Vec6f>& trianglesList, unordered_set<Cluster *>& clusters, Size imgSize)
{
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






void findSimilars(unordered_set<Cluster *> clusters, Cluster * c, unordered_set<Cluster *>& similars, double threshold) {

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


vector<int> generateIndexArray(vector<Cluster * > clusters) {
	vector<int> toRet;
	for (int i = 0; i < clusters.size(); i++) {
		toRet.push_back(i);
	}
	return toRet;
}

vector<Cluster*> indexArrayToClusterArray(vector<int> indexArray, vector<Cluster*> base) {
	vector<Cluster *> toRet;
	for each (int i in indexArray)
	{
		toRet.push_back(base.at(i));
	}
	return toRet;
}

vector<vector<int>> nChooseKPermutations(vector<int> arr, int k) {
	vector<vector<int>> combs;
	vector<int> vec;
	nChooseKPermutationsRec(0, k,arr,combs,vec);
	return combs;
}


void nChooseKPermutationsRec(int offset, int k, vector<int> arr,vector<vector<int>> &combs,vector<int> &current) {
	if (k == 0) {
		vector<int> newVec(current);
		combs.push_back(newVec);
		return;
	}
	for (int i = offset; i <= arr.size() - k; ++i) {
		current.push_back(arr[i]);
		nChooseKPermutationsRec(i + 1, k - 1,arr,combs,current);
		current.pop_back();
	}
}

Cluster * findBestCluster(Cluster * c,vector<Cluster*> clusters, unordered_set<int> ignore ,double clustersDistanceTreshold, double maxQueryDist) {
	double minD = INF;
	Cluster * toRet=NULL;

	for each (Cluster* p in clusters)
	{
		if (p != c && ignore.find((int)p)==ignore.end()) {
			double d = euclideanDist(c->getClusterCenterPoint(), p->getClusterCenterPoint());
			if (d <= maxQueryDist) {
				double dist = compareHist(c->getClusterCenterDescriptor(), p->getClusterCenterDescriptor(), CV_COMP_CHISQR);
				if (dist <= clustersDistanceTreshold) {
				
				if (dist < minD) {
					minD = dist;
					toRet = p;
				}
				}
			}
		}
		

	}
	return toRet;
}

vector<vector<Cluster*> > permGenerator(vector<Cluster *> clusters,int k)
{
	vector<int> indexVector = generateIndexArray(clusters);
	vector<vector<int>> combs = nChooseKPermutations(indexVector, k);
	vector<vector<Cluster*>> combinations;

	for each (vector<int> v in combs)
	{
		combinations.push_back(indexArrayToClusterArray(v, clusters));
	}
	return combinations;
}

vector<vector<Cluster*>> match(vector<Cluster*> page, vector<Cluster*> query, double clustersDistanceTreshold, double maxQueryDist) {
	unordered_set<int> ignore;
	vector<Cluster *> best;
	vector< vector<Cluster*> > container;
	for each (Cluster * qC in query)
	{
		
		Cluster * found=findBestCluster(qC, page, ignore, clustersDistanceTreshold, maxQueryDist);
		if (found != NULL) {
			best.push_back(found);
			ignore.insert((int)found);
		}
		
	}
	container.push_back(best);
	return container;
}

//vector<vector<Cluster*>> match(vector<Cluster*> page,vector<Cluster*> query,double clustersDistanceTreshold,double maxQueryDist) {
//	double dist = -INFINT;
//	bool validPoint=true;
//	vector<vector<Cluster*>> meetsRequierments;
//	vector<vector<Cluster*>> perms=permGenerator(page, query.size());
//	cout << "perm size: " << perms.size() << endl;
//	double minD = INF;
//	for each (vector<Cluster*> p in perms)
//	{
//		double d = calculateClusterMaxDistance(p);
//		if (d <= maxQueryDist) {
//			double dist = calculateMinDistanceBetweenKClusters(p, query);
//			if (dist <= clustersDistanceTreshold) {
//				meetsRequierments.push_back(p);
//			}
//			if (dist < minD) {
//				minD = dist;
//			}
//		}
//		
//	}
//	cout << "min dist is:" << minD << endl;
//
//	return meetsRequierments;
//}

double calculateMinDistanceBetweenKClusters(vector<Cluster *>page,vector<Cluster*> query) {
	sort(page.begin(), page.end());
	double min = INF;
	do
	{
		double d=calculateDistanceBetweenKClusters(page, query);
		if (d < min) {
			min = d;
		}

	} while (next_permutation(page.begin(), page.end()));
	return min;

}

double calculateClusterMaxNeighborDistance(unordered_set<Cluster * > clusters) {
	double max = -INF;
	for each (Cluster * c1 in clusters)
	{
		for each (Cluster * c2 in c1->getNeighbours()) {
			if (c1 != c2) {
				double d = euclideanDist(c1->getClusterCenterPoint(), c2->getClusterCenterPoint());
				max = MAX(d, max);
			}
		}
	}
	return max;
}

double calculateClusterMaxDistance(vector<Cluster*> clusters) {
	double max = -INF;
	for each (Cluster * c1 in clusters)
	{
		for each (Cluster * c2 in clusters) {
			if (c1 != c2) {
				double d=euclideanDist(c1->getClusterCenterPoint(), c2->getClusterCenterPoint());
				max=MAX(d, max);
			}
		}
	}
	return max;
}

double calculateDistanceBetweenKClusters(vector<Cluster *>page, vector<Cluster*> query) {
	double total = 0;
	int i = 0;
	for each (Cluster * c in page)
	{
		total += compareHist(c->getClusterCenterDescriptor(), query.at(i)->getClusterCenterDescriptor(),CV_COMP_CHISQR);
		
	}
	return total;
}


pair<double, unordered_set<Cluster *>> buildClusters(Mat img,  double vectorDirectionThreshold,double distance) {
	//cout << vectorDirectionThreshold << endl;
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();
	vector<KeyPoint> keypoints;
	Mat descriptors;
	sft->detect(img, keypoints);
	sft->compute(img, keypoints, descriptors);
	unordered_set<Cluster *> clusters;
	//unordered_set<Cluster *> joinedClusters;
	initClusters(keypoints, descriptors, clusters);
	//calculateTrigularityList(clusters,img.size());
	vector<Vec6f> trianglesList;
	calcTriangularityList (trianglesList,clusters, img.size());
	setClusterNeighboursByTriangles (trianglesList,clusters, img.size());
	double maxDis;
	maxDis=joinClustersFixedPoint(clusters,vectorDirectionThreshold,img.size(), img, distance);

	//for each (Cluster * c in clusters){
	//	for each (Cluster * n in c->getNeighbours()) 
	//		line(img, c->getClusterCenterPoint() , n->getClusterCenterPoint() ,  Scalar( 100, 100, 100 ));
	//}

	//imshow("lines.jpg",img);

	//waitKey();

	return make_pair(maxDis, clusters);

}



double joinClustersFixedPoint(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, Size imgSize,Mat img, double optionalDistance){
//double joinClustersFixedPoint(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold,double optionalDistance) {

	double maxDistance = -1;
	int previousCulstersSize = -1;

	maxDistance=max(maxDistance, joinClusters(clusters, vectorDirectionThreshold,imgSize,img));
	cout << "prev: " << previousCulstersSize  << " curr: " << clusters.size() << endl;
	while (clusters.size() != previousCulstersSize)
	{
		previousCulstersSize = clusters.size();
		maxDistance=max(maxDistance, joinClusters(clusters, vectorDirectionThreshold,imgSize,img));
		cout << "prev: " << previousCulstersSize  << " curr: " << clusters.size() << endl;
	}
	
	maxDistance = calculateClusterMaxNeighborDistance(clusters);

	

	return maxDistance;

}


//double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, double optionalDistance) {
//
//
//	unordered_set<Cluster *> handled;
//	unordered_set<Cluster *> toRemove;
//	double maxDistance = -1;
//	for each(Cluster * c in clusters) {
//		unordered_set<Cluster *> neighbours = c->getNeighbours();
//		handled.insert(c);
//		for each(Cluster * n in neighbours) {
//			if (handled.find(n) != handled.end() || !mergable(c, n, vectorDirectionThreshold, optionalDistance)) {
//				continue;
//			}
//			double dist = euclideanDist(c->getClusterCenterPoint(), n->getClusterCenterPoint());
//			maxDistance = max(dist, maxDistance);
//			c->mergeClusters(n);
//			toRemove.insert(n);
//
//		}
//	}
//		for each (Cluster *c in toRemove)
//		{
//			clusters.erase(c);
//		}
//
//		return maxDistance;
//}



//double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, Size imgSize, double optionalDistance){
double joinClusters(unordered_set<Cluster *>& clusters, double vectorDirectionThreshold, Size imgSize,Mat img, double optionalDistance){
	//unordered_set<Cluster *> handled;
	unordered_set<Cluster *> toRemove;
	double maxDistance = -1;
	unordered_set<int> visited;
	for ( auto it = clusters.begin(); it != clusters.end(); ++it ){
		
    	Cluster* c = *it;
		int val = (int)c;
		if (visited.find(val) != visited.end()) {
			cout << "visted called" << endl;
			continue;
		}
		unordered_set<Cluster *> neighbours = c->getNeighbours();
		//handled.insert(c);
		for each(Cluster * n in neighbours) {
			if (!mergable(c, n, vectorDirectionThreshold, optionalDistance)) {
				continue;
			}
			double dist = euclideanDist(c->getClusterCenterPoint(), n->getClusterCenterPoint());
			maxDistance = max(dist, maxDistance);
			c->mergeClusters(n);
			toRemove.insert(n);
			visited.insert((int)n);
		}
		

		//calculateTrigularityList(clusters,  imgSize);
		//vector<Vec6f> trianglesList;
		//calcTriangularityList (trianglesList,clusters, img.size());
		//setClusterNeighboursByTriangles (trianglesList,clusters, img.size());

	}
	for each (Cluster *c in toRemove)
	{
		clusters.erase(c);
	}
		return maxDistance;
}





pair<double,double> calculateDirection(Vec<double, 128> v) {
	
	Vec<double,8> dirSum;
	int entry;
	for (int i = 0; i <128; i++) {
		entry = i % 8;
		dirSum[entry] += v[i];
	}
	int max = 0;
	for (int i = 0; i < 8; i++) {
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
		Vec<double, 128> v;
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


vector<double> getIdentificationVector(Mat image, int gridWidth, int gridHeight) {
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
	vector<double> grid;
	for (int i = 0; i < gridWidth; i++) {
		for (int j = 0; j < gridHeight; j++) {
			Mat m = gridMultiplePoints.at(i).at(j);
			if (m.empty()) {
				m = Mat::zeros(Size(128, 1), CV_32F);
			}
			Vec<double, 128> d = calculateGrid(m);
			grid.insert(end(grid), begin(d.val), end(d.val));

		}
	}
	grid.shrink_to_fit();
	return grid;





}

Vec<double, 128> calculateGrid(Mat cell) {
	if (cell.rows < 2) {
		return cell;
	}
	reduce(cell, cell, 0, CV_REDUCE_AVG, -1);
	return cell;

}

double compare(Vec<double,128> a, Vec<double,128> b) {
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