#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstring>           // strcmp()
#include <vector>            // std::vector
#include <cuda_runtime.h>
#include <chrono>
using namespace std::chrono;
#include <inttypes.h>
#include <iostream>

#ifdef BOINC
constexpr int RUNS_PER_CHECKPOINT = 16;
#include "boinc/boinc_api.h"
#if defined _WIN32 || defined _WIN64
#include "boinc/boinc_win.h"
#endif
#endif

// for boinc checkpointing
struct checkpoint_vars {
    uint64_t range_min;
    uint64_t range_max;
    uint32_t stored_checksum;
    uint64_t elapsed_chkpoint;
};

// Constants
constexpr unsigned long long THREAD_SIZE = 512;
constexpr unsigned long long BLOCK_SIZE = 1ULL << 23;
constexpr unsigned long long BATCH_SIZE = BLOCK_SIZE * THREAD_SIZE;
constexpr int RESULTS_BUFFER_SIZE = 8;
constexpr int SCORE_CUTOFF = 50;

constexpr int HASH_BATCH_SIZE = 4; // (N+1)/N = 1.25 splitmix64 per setSeed, vs 2.0 in regular bruteforce
constexpr uint64_t XL = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t XH = 0x6A09E667F3BCC909ULL;
constexpr uint64_t XL_BASE = XL * HASH_BATCH_SIZE;

// Structs
struct Result {
    int64_t  score;
    uint64_t seed;
    int64_t  a, b;
};

// Xoroshiro impl
__device__ __forceinline__ uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    const uint64_t M1 = 0xBF58476D1CE4E5B9ULL;
    const uint64_t M2 = 0x94D049BB133111EBULL;
    z = (z ^ (z >> 30)) * M1;
    z = (z ^ (z >> 27)) * M2;
    return z ^ (z >> 31);
}

struct PRNG128 {
    uint64_t lo, hi;

    // __device__ explicit PRNG128(uint64_t world_seed) {
    //     const uint64_t XH = 0x6A09E667F3BCC909ULL;
    //     const uint64_t XL = 0x9E3779B97F4A7C15ULL;
    //     uint64_t s = world_seed ^ XH;
    //     lo = mix64(s);
    //     hi = mix64(s + XL);
    // }

    __device__ explicit PRNG128(uint64_t s) {
        lo = mix64(s);
        hi = mix64(s + XL);
    }

    __device__ explicit PRNG128(uint64_t _lo, uint64_t _hi) {
        lo = _lo;
        hi = _hi;
    }

    __device__ uint64_t next64() {
        uint64_t res = rotl64(lo + hi, 17) + lo;
        uint64_t t   = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
        return res;
    }

    __device__ uint32_t nextLongLower32() {
        uint64_t t = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
        t = hi ^ lo;
        return static_cast<uint32_t>((rotl64(lo + hi, 17) + lo) >> 32);
    }

    __device__ void advance() {
        uint64_t t = hi ^ lo;
        lo = rotl64(lo, 49) ^ t ^ (t << 21);
        hi = rotl64(t, 28);
    }

    __device__ int64_t nextLong() {
        int32_t high = static_cast<int32_t>(next64() >> 32);
        int32_t low  = static_cast<int32_t>(next64() >> 32);
        return (static_cast<int64_t>(high) << 32) + static_cast<int64_t>(low);
    }
};

__device__ __forceinline__ void compute_ab(uint64_t seed, int64_t &a, int64_t &b) {
    PRNG128 rng(seed);
    a = rng.nextLong() | 1LL;
    b = rng.nextLong() | 1LL;
}

// Utils
static void gpuAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d — %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) gpuAssert((ans), __FILE__, __LINE__)

// Computation

__device__ static inline bool goodLower32(PRNG128& rng) {
    uint32_t al = rng.nextLongLower32() | 1U;
    rng.advance();
    uint32_t bl = rng.nextLongLower32() | 1U;

    return 
        al == bl || al + bl == 0 ||
        3*al == bl || 3*al + bl == 0 ||
        al == 3*bl || al + 3*bl == 0 ||
        5*al == bl || 5*al + bl == 0 ||
        al == 5*bl || al + 5*bl == 0 ||
        3*al == 5*bl || 3*al + 5*bl == 0 ||
        5*al == 3*bl || 5*al + 3*bl == 0 ||
        7*al == bl || 7*al + bl == 0 ||
        al == 7*bl || al + 7*bl == 0 ||
        7*al == 3*bl || 7*al + 3*bl == 0 ||
        7*al == 5*bl || 7*al + 5*bl == 0 ||
        5*al == 7*bl || 5*al + 7*bl == 0 ||
        7*al == 3*bl || 7*al + 3*bl == 0;
}

__device__ static inline void processFullPrngState(uint64_t xseed, Result *results, volatile int *result_idx) {  
    // score >= 32, vast majority of cases eliminated already
    // now we can just do a full check

    int64_t a, b;
    compute_ab(xseed, a, b);

    int64_t score = 0;
    uint64_t x;
    int tz;

    x = (uint64_t)a ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)a ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-a) ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(3 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-3 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)a ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-a) ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(5 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-5 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(3 * a) ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-3 * a) ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(5 * a) ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-5 * a) ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)a ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-a) ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(7 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-7 * a) ^ (uint64_t)b;
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(3 * a) ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-3 * a) ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(7 * a) ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-7 * a) ^ (uint64_t)(3 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(5 * a) ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-5 * a) ^ (uint64_t)(7 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(7 * a) ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    x = (uint64_t)(-7 * a) ^ (uint64_t)(5 * b);
    tz = x ? (__ffsll(x) - 1) : 64;
    score = tz > score ? tz : score;

    if (score < SCORE_CUTOFF)
        return;

    uint64_t seed = xseed ^ XH; // going back from xSetSeed internal value
    int this_result_idx = atomicAdd((int*)result_idx, 1);
    results[this_result_idx] = { score, seed, a, b };
}

// each thread checks HASH_BATCH_SIZE seeds
__global__ void searchKernel(uint64_t start_seed, Result *results, volatile int *result_idx, volatile uint32_t *checksum) {
    uint64_t gid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t seed_base = (start_seed + gid) * XL_BASE;

    uint64_t hashes[HASH_BATCH_SIZE + 1];
    #pragma unroll
    for (int i = 0; i <= HASH_BATCH_SIZE; i++)
        hashes[i] = mix64(seed_base + i*XL);

    #pragma unroll
    for (int i = 0; i < HASH_BATCH_SIZE; i++) {
        PRNG128 prng{hashes[i], hashes[i+1]};
        if (!goodLower32(prng))
            continue;
        uint64_t curr_s = seed_base + i * XL;
        processFullPrngState(curr_s, results, result_idx);
	    atomicAdd((uint32_t*)checksum, 1);
    }
}

int main(int argc, char **argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t time_elapsed = 0;
    uint64_t start_seed = 0;
    uint64_t original_start_seed = 0;
    uint64_t end_seed   = 211414360895ULL / HASH_BATCH_SIZE;
    uint64_t device_id  = 0;
    uint32_t *checksum;
    FILE* seed_output = fopen("seeds.txt", "a");
    int ret =  cudaMallocManaged(&checksum, sizeof(uint32_t));
    *checksum = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc){
            start_seed = strtoull(argv[++i], nullptr, 0);
            original_start_seed = start_seed;
        }

        else if (!strcmp(argv[i], "-e") && i + 1 < argc)
            end_seed = strtoull(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc)
            device_id = strtoull(argv[++i], nullptr, 0);
        else {
            printf("Usage: %s [-s start_seed] [-e end_seed] [-d device_id]\n", argv[0]);
            return 0;
        }
    }
    #ifdef BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true;
        boinc_init_options(&options);
        APP_INIT_DATA aid;
        boinc_get_init_data(aid);
        if (aid.gpu_device_num >= 0) {
            //If BOINC client provided us a device ID
            device_id = aid.gpu_device_num;
            fprintf(stderr, "boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, device_id);
        }
        else {
            //If BOINC client did not provide us a device ID
            device_id = -5;
            for (int i = 1; i < argc; i += 2) {
                //Check for a --device flag, just in case we missed it earlier, use it if it's available. For older clients primarily.
                if (strcmp(argv[i], "-d") == 0) {
                    sscanf(argv[i + 1], "%i", &device_id);
                }

            }
            if (device_id == -5) {
                //Something has gone wrong. It pulled from BOINC, got -1. No --device parameter present.
                fprintf(stderr, "Error: No --device parameter provided! Defaulting to device 0...\n");
                device_id = 0;
            }
            fprintf(stderr, "stndalone gpuindex %i (aid value: %i)\n", device_id, aid.gpu_device_num);
        }
        FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
        if (!checkpoint_data) {
            //No checkpoint file was found. Proceed from the beginning.
            fprintf(stderr, "No checkpoint to load\n");
        }
        else {
            //Load from checkpoint. You can put any data in data_store that you need to keep between runs of this program.
            boinc_begin_critical_section();
            struct checkpoint_vars data_store;
            fread(&data_store, sizeof(data_store), 1, checkpoint_data);
            start_seed = data_store.range_min;
            end_seed = data_store.range_max;
            time_elapsed = data_store.elapsed_chkpoint;
            *checksum = data_store.stored_checksum;
            fprintf(stderr, "Checkpoint loaded, task time %llu us, seed pos: %llu, checksum val: %llu\n", time_elapsed, start_seed, *checksum);
            fclose(checkpoint_data);
            boinc_end_critical_section();
        }
    #endif // BOINC

    CUDA_CHECK(cudaSetDevice(device_id));

    Result *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, RESULTS_BUFFER_SIZE * sizeof(Result)));
    
    Result h_results[RESULTS_BUFFER_SIZE];

    int *results_count;
    cudaMallocManaged(&results_count, sizeof(int));



    cudaDeviceSynchronize();
    uint64_t checkpointTemp = 0;
    for (uint64_t curr_seed = start_seed; curr_seed <= end_seed; curr_seed += BATCH_SIZE) {
        *results_count = 0;

        searchKernel<<<BLOCK_SIZE, THREAD_SIZE>>>(curr_seed, d_results, results_count, checksum);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        printf("After batch %llu: checksum = %u\n", curr_seed, *checksum);
        #ifdef BOINC
            if (checkpointTemp >= RUNS_PER_CHECKPOINT-1 || boinc_time_to_checkpoint()) {
                cudaDeviceSynchronize();
                auto checkpoint_time = std::chrono::high_resolution_clock::now();
                time_elapsed = duration_cast<milliseconds>(checkpoint_time - start_time).count() + time_elapsed;
                //Checkpointing for BOINC
                boinc_begin_critical_section(); // Boinc should not interrupt this

                checkpointTemp = 0;
                boinc_delete_file("checkpoint.txt"); // Don't touch, same func as normal fdel
                FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "wb");

                struct checkpoint_vars data_store;
                data_store.range_min = curr_seed + BATCH_SIZE; // this seed was already completed, processing can resume from next seed
                data_store.range_max = end_seed;
                data_store.elapsed_chkpoint = time_elapsed;
                data_store.stored_checksum = *checksum;
                fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
                fclose(checkpoint_data);

                boinc_end_critical_section();
                boinc_checkpoint_completed(); // Checkpointing completed
            }
            // Update boinc client with percentage
            double frac = (double)(curr_seed - start_seed + 1) / (double)(end_seed - start_seed);
            boinc_fraction_done(frac);
        #endif // BOINC
        if (*results_count > 0) {
            CUDA_CHECK(cudaMemcpy(&h_results, d_results, *results_count * sizeof(Result), cudaMemcpyDeviceToHost));

            for (uint64_t i = 0; i < *results_count; i++) {
                Result result = h_results[i];
                //std::cout << "seed: " << result.seed << " score: " << result.score << std::endl;
                fprintf(seed_output, "seed: %lld score: %lld\n", result.seed, result.score);
                //std::printf("seed: %" PRIu64 " score: %" PRId64 " (a=%I64x b=%I64x)\n", result.seed, result.score, result.a, result.b);
            }
        }
        checkpointTemp++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    fprintf(seed_output, "Checksum: %lld\n", *checksum);
    fprintf(stderr, "Seeds checked: %lld\n", (end_seed - original_start_seed) * HASH_BATCH_SIZE );
    fprintf(stderr, "Time taken: %lldms\n", duration.count()+time_elapsed);
    fprintf(stderr, "GSPS: %lld\n", (end_seed - original_start_seed) * HASH_BATCH_SIZE / duration.count()+time_elapsed * 10e3 / 10e9);
    // std::cout << "Checksum: " << *checksum << std::endl;
    // std::cout << "Seeds checked: " << (end_seed - start_seed) * HASH_BATCH_SIZE << std::endl;
    // std::cout << "Time taken: " << duration.count()+(double)time_elapsed << "ms" << std::endl;
    // double sps = (end_seed - start_seed) * HASH_BATCH_SIZE / duration.count() * 10e3 / 10e9;
    // std::cout << "GSPS: " << sps << std::endl;

    #ifdef BOINC
        boinc_finish(0);
    #endif
}