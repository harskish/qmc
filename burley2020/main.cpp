#include <cstdlib>
#include <vector>
#include <stdio.h>
#include "genpoints.h"


#include <torch/extension.h>
#include <iostream>

void help()
{
    printf("Usage: genpoints [seq] [N=16] [dim=0] [seed=1]\n"
           "seq is one of:\n"
           "   random\n"
           "   sobol\n"
           "   sobol_rds\n"
           "   sobol_owen\n"
           "   laine_karras\n"
           "   faure05\n");
 exit(1);
}

int main(int argc, const char** argv)
{
    argc--; argv++;

    if (argc < 1) help();
    const char* seq = *argv++;
    int n = argc < 2 ? 16 : std::atoi(*argv++);
    int dim = argc < 3 ? 0 : std::atoi(*argv++);
    int seed = argc < 4 ? 1 : std::atoi(*argv++);

    if (n < 0) n = 0;
    if (dim < 0 || dim > 4) dim = 0;

    std::vector<float> x(n);
    genpoints(seq, n, dim, seed, x.data());
    for (int i = 0; i < n; i++) {
        printf("%g\n", x[i]);
    }
    return 0;
}

std::vector<float> sample(const char* seq, int n, int dim, const int seed) {
    if (n < 0) n = 0;
    if (dim < 0 || dim > 4) dim = 0;
    std::vector<float> x(n);
    genpoints(seq, n, dim, seed, x.data());
    return x;
}

PYBIND11_MODULE(burley2020ext, m) {
    //m.def("register", &register_resource, "register resource");
    //m.def("unregister", &unregister_resource, "unregister resource");
    //m.def("upload", &upload, "upload image data");
    m.def("sample", &sample, "sample sequence");
}
