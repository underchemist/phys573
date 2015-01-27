// Copyright 2015 Yann-Sebastien Tremblay-Johnston
#include "./qmc.hpp"
#include "./mt.hpp"

void init(walkers &r) {
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

double psi(const coord &c1, const coord &c2) {
    double r1 = 0.0;
    double r2 = 0.0;
    double r12 = 0.0;
    double p;

    for (size_t i = 0; i < DIM; i++) {
        r1 += c1[i] * c1[i];
        r2 += c2[i] * c2[i];
        r12 += (c1[i] - c2[i]) * (c1[i] -c2[i]);
    }

    r1 = sqrt(r1);
    r2 = sqrt(r2);
    r12 = sqrt(r12);
    p = -2.0 * r1 - 2.0 * r2 + r12 / (2.0 * (1.0 + alpha * r12));

    return exp(p);
}

double elocal(const coord &c1, const coord &c2) {
    double r1 = 0.0;
    double r2 = 0.0;
    double r12 = 0.0;
    double dprod = 0.0;
    double d1;
    double d2;
    double d3;
    double d4;

    // compute norms
    for (size_t i = 0; i < DIM; i++) {
        r1 += c1[i] * c1[i];
        r2 += c2[i] * c2[i];
        r12 += (c1[i] - c2[i]) * (c1[i] -c2[i]);
    }

    r1 = sqrt(r1);
    r2 = sqrt(r2);
    r12 = sqrt(r12);

    // compute dot product
    for (size_t i = 0; i < DIM; i++) {
        dprod += ((c1[i] - c2[i]) / r12) * ((c1[i] / r1) - (c2[i] / r2));
    }

    d1 = 1.0 / (1.0 + alpha * r12);
    d2 = d1 * d1;
    d3 = d2 * d1;
    d4 = d2 * d2;

    return -4.0 + alpha * (d1 + d2 + d3) - d4 / 4.0 + dprod * d2;
}

int main() {
    walkers r;

    // initialize walkers with positions in [0, 1)
    init(r);

    for (size_t i = 0; i < N; i++) {
        cout << elocal(r[i][0], r[i][1]) << endl;
    }
    return 0;
}
