// Copyright 2015 Yann-Sebastien Tremblay-Johnston
#include "./qmc.h"

int main() {
    typedef boost::mt19937 RNGType;    // Mersenne Twister

    RNGType  rng(time(0));
    boost::random::uniform_real_distribution< > dist(-10.0, 10.0);
    boost::variate_generator<RNGType,
        boost::random::uniform_real_distribution< > >
        gen(rng, dist);

    for (int i = 0; i < 10; i++)
        cout << gen() << endl;

    return 0;
}
