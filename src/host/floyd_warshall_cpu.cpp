#include "floyd_warshall_cpu.h"
#include <iostream>
#include <vector>

using namespace std;

void floydWarshallCPU(vector<float>& graph, int n) {
    cout << "Running Floyd-Warshall on CPU" << endl;
    for (int k = 0; k < n; k++)
    {
        cout << k + 1 << "/" << n << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (graph[i * n + j] > graph[i * n + k] + graph[k * n + j])
                {
                    graph[i * n + j] = graph[i * n + k] + graph[k * n + j];
                }
            }
        }
    }
}