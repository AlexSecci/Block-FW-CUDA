#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

inline void writeGraphToFile(const vector<float>& graph, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    for (const auto& val : graph) {
        outFile << val << "\n";
    }
    outFile.close();
    cout << "Graph successfully written to " << filename << endl;
}

inline void verifyResults(vector<float>& graphCPU, vector<float>& graphGPU) {
    // TODO: Compare matrices
    cout << "Verifying results..." << endl;
    for (size_t i = 0; i < graphCPU.size(); i++)
    {
        if (graphCPU[i] != graphGPU[i]){
            cout << "Error found at index " << i << "." << endl;
            cout << "Expected: " << graphCPU[i] << ", Found: " << graphGPU[i] << endl;
            return;
        }
    }
    cout << "No errors found" << endl;
}


#endif // UTILS_H
