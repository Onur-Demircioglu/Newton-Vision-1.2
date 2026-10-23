// C++ çekirdek self-test — python_ref/test_ref.py ile aynı elle-hesaplanır
// değerleri denetler. Bu bir DOĞRULUK testidir, hız değil.
// Doğrulamadan pybind11'e bağlama.
#include "layers.h"

#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

using namespace nv;

static int g_fail = 0;

template <typename T>
static void check_eq(const std::string& name,
                     const std::vector<T>& got, const std::vector<T>& want) {
    bool ok = (got.size() == want.size());
    for (size_t i = 0; ok && i < got.size(); ++i)
        if (std::fabs(float(got[i]) - float(want[i])) > 1e-5f) ok = false;
    if (!ok) { g_fail++;
        std::printf("FAIL %s\n got:  ", name.c_str());
        for (auto v : got) std::printf("%g ", v);
        std::printf("\n want: ");
        for (auto v : want) std::printf("%g ", v);
        std::printf("\n");
    } else {
        std::printf("PASS %s  [", name.c_str());
        for (auto v : got) std::printf("%g ", v);
        std::printf("]\n");
    }
}

static void test_conv_small() {
    // test_ref.py ile birebir: x=1..9 (3x3), w=all-1 (2x2), stride1 pad0 bias0
    // Beklenen: [12,16,24,28]
    float x[] = {1,2,3, 4,5,6, 7,8,9};
    float w[] = {1,1, 1,1};
    float bias0[] = {0.0f};
    int Ho, Wo;
    auto y = conv2d(x, {1,1,3,3}, w, 1, 2, 2, bias0, /*stride*/1, /*padding*/0, /*dilation*/1, /*groups*/1, Ho, Wo);
    if (Ho != 2 || Wo != 2) { g_fail++; std::printf("FAIL conv_small boyut: %dx%d\n", Ho, Wo); }
    check_eq("conv_small(all1)", y, std::vector<float>{12,16,24,28});
}

static void test_relu_small() {
    // test_ref.py: x=[-2,-1,0,1,2]
    float x[] = {-2,-1,0,1,2};
    auto y = relu(x, 5);
    check_eq("relu_small", y, std::vector<float>{0,0,0,1,2});
}

static void test_maxpool_small() {
    // test_ref.py ile birebir: x=0..15 (4x4), k=2 s=2
    // Blok max'ları: [0,1,4,5]->5, [2,3,6,7]->7, [8,9,12,13]->13, [10,11,14,15]->15
    float x[] = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};
    int Ho, Wo;
    auto y = maxpool2d(x, {1,1,4,4}, 2, 2, Ho, Wo);
    if (Ho != 2 || Wo != 2) { g_fail++; std::printf("FAIL maxpool boyut: %dx%d\n", Ho, Wo); }
    check_eq("maxpool_small", y, std::vector<float>{5,7,13,15});
}

int main() {
    test_conv_small();
    test_relu_small();
    test_maxpool_small();
    if (g_fail) { std::printf("\n%d test FAILED\n", g_fail); return 1; }
    std::printf("\nALL PASS\n");
    return 0;
}
