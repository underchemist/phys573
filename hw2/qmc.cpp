// Copyright 2015 Yann-Sebastien Tremblay-Johnston
#include "./qmc.h"
#include "./mt.h"

const size_t DIM = 3;  // dimension
const size_t NEL = 2;  // number of electrons for He
const size_t N = 500;  // number of random walkers


void init(vector<vector<vector<double> > > &r) {
    r.resize(N);
    for (size_t i = 0; i < N; i++) {
        r[i].resize(NEL);
        for (size_t j = 0; j < NEL; j++) {
            r[i][j].resize(DIM);
            for (size_t k = 0; k < DIM; k++) {
                r[i][j][k] = rnd_gen();
            }
        }
    }
}

int main() {
    // array of walkers, each a pair of 2 electrons with 3 coord
    vector<vector<vector<double> > > r;

    init(r);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < NEL; j++) {
            cout << "walker: " << i + 1
                     << " electron: " << j + 1
                     << " x = " << r[i][j][0]
                     << " y = " << r[i][j][1]
                     << " z = " << r[i][j][2] << endl;
        }
    }
    return 0;
}
