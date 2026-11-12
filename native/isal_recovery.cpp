// Recovery from already surviving shards; ISA-L arithmetic stays unchanged.
#include <erasure_code.h>
#include <cstring>
#include <vector>

extern "C" int requant_isal_recover(int k, int rows, int size, int threads,
    unsigned char* tables, const int* map, unsigned char* survivors,
    unsigned char* recovered, unsigned char* restored) {
    if (k <= 0 || rows <= 0 || size <= 0 || threads < 1 || threads > 12 ||
        !tables || !map || !survivors || !recovered || !restored) return -1;
    for (int i = 0; i < k; ++i)
        if (map[i] >= k || map[i] < -rows) return -1;
    auto chunk = [&](int offset, int length) {
        std::vector<unsigned char*> source(k), output(rows);
        for (int i = 0; i < k; ++i) source[i] = survivors + (size_t)i * size + offset;
        for (int i = 0; i < rows; ++i) output[i] = recovered + (size_t)i * size + offset;
        ec_encode_data(length, k, rows, tables, source.data(), output.data());
        for (int i = 0; i < k; ++i)
            std::memcpy(restored + (size_t)i * size + offset,
                        map[i] >= 0 ? source[map[i]] : output[-map[i]-1], length);
    };
    if (threads == 1) chunk(0, size);
    else {
        constexpr int step = 262144;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int offset = 0; offset < size; offset += step)
            chunk(offset, offset + step <= size ? step : size - offset);
    }
    return 0;
}
