#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

class Graph
{
	public:
		map<int, bool> visited;
		map<int, list<int>> adj;

		// function to add an edge to graph
		void addEdge(int v, int w);

		// DFS traversal of the vertices reachable from v
		void DFS(int v);
};

void Graph::addEdge(int v, int w)
{
	// Add w to v’s list.
	adj[v].push_back(w);
}

void Graph::DFS(int v)
{
#pragma omp parallel
	// Mark the current node as visited and print it
	cout << "Visited: " << v << " by Thread " << omp_get_thread_num() << endl;

	visited[v] = true;
	cout << v << " ";

	// Recur for all the vertices adjacent to this vertex
	list<int>::iterator i;
	for (i = adj[v].begin(); i != adj[v].end(); ++i)
	{
		if (!visited[*i])
			DFS(*i);
	}
}

int main()
{
	int z;
	Graph g;
	omp_set_num_threads(4);
	g.addEdge(0, 1);
	g.addEdge(0, 2);
	g.addEdge(1, 3);
	g.addEdge(2, 3);
	g.addEdge(3, 4);
	g.addEdge(3, 5);
	g.addEdge(2, 6);

	cout << "Enter the vertex to start the DFS traversal with: " << endl;
	cin >> z;

	cout << "\nDepth First Traversal: \n";
	g.DFS(z);
	cout << endl;

	return 0;
}


//output

// Enter the vertex to start the DFS traversal with:
// 0

// Depth First Traversal:
// Visited: 0 by Thread 0
// 0 Visited: 1 by Thread 2
// Visited: 2 by Thread 3
// Visited: 3 by Thread 1
// 1 3 Visited: 4 by Thread 2
// Visited: 5 by Thread 1
// 4 Visited: 6 by Thread 0
// 5 6 2
