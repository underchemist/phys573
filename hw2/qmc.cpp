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

int main() {
    walkers r;

    // initialize walkers with positions in [0, 1)
    init(r);

    for (size_t i = 0; i < N; i++) {
        cout << psi(r[i][0], r[i][1]) << endl;
    }
    return 0;
}
