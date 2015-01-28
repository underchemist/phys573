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

void adjust_delta() {
    delta *= accept_count / (0.5 * N);
    accept_count = 0;
}

void mstep(electrons &e) {
    coord c1(e[0]);
    coord c2(e[1]);
    coord t1(DIM, 0.0);
    coord t2(DIM, 0.0);
    double w;
    double E;

    for (size_t i = 0; i < DIM; i++) {
        t1[i] = c1[i] + delta * (2.0 * rnd_gen() - 1.0);
        t2[i] = c2[i] + delta * (2.0 * rnd_gen() - 1.0);
    }

    // move
    w = psi(t1, t2) / psi(c1, c2);
    w *= w;

    if (rnd_gen() < w) {
        for (size_t i = 0; i < DIM; i++) {
            e[0][i] = t1[i];
            e[1][i] = t2[i];
        }
        ++accept_count;
    }
    E = elocal(e[0], e[1]);
    Esum += E;
    Esum_sqr += E * E;
}

void singleMCstep(walkers &r) {
    for (size_t i = 0; i < N; i++) {
        mstep(r[i]);
    }
}

void MCtherm(walkers &r) {
    for (int i = 0; i < thermsteps; i++) {
        singleMCstep(r);
        adjust_delta();
    }
    Esum = 0.0;
    Esum_sqr = 0.0;
}

void MCrun(walkers &r) {
    for (int i = 0; i < MCsteps; i++) {
        singleMCstep(r);
        adjust_delta();
    }
}

int main() {
    walkers r;
    double Eave;
    double Evar;
    double Err;
    ofstream f;
    f.open("data.csv");
    f << "Eave,Evar,Err,alpha" << endl;

    // initialize walkers with positions in [0, 1)
    init(r);

    for (int i = 0; i <= 100; i++) {
        MCtherm(r);

        MCrun(r);

        Eave = Esum / double(N) / double(MCsteps);
        Evar = Esum_sqr / double(N) / double(MCsteps) - Eave * Eave;
        Err = sqrt(Evar) / sqrt(double(N) * double(MCsteps));
        cout << "alpha = " << alpha
             << " <E> = " << Eave
             << " +/- " << Err
             << " <Evar> = " << Evar << endl;

        f << Eave << "," << Evar << "," << Err << "," << alpha << endl;
        alpha += 0.005;

    }
    f.close();

    return 0;
}
