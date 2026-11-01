// CUDA_INJECTION64_PATH library: one CSV line per kernel / memcpy, via the
// CUPTI activity API (timestamps only, so no performance-counter permission).
//
//   K,start_ns,end_ns,stream,graph_id,correlation,grid,block,name
//   M,start_ns,end_ns,stream,graph_id,correlation,copy_kind,bytes
//
// Kernels of one CUDA-graph replay share the correlation id of the
// cudaGraphLaunch, which is how a trace is cut into decode steps.
#include <cupti.h>

#include <cstdio>
#include <cstdlib>

static FILE *out = nullptr;

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_records) {
    *size = 16u << 20;
    *buffer = static_cast<uint8_t *>(aligned_alloc(8, *size));
    *max_records = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream, uint8_t *buffer,
                                      size_t, size_t valid) {
    CUpti_Activity *record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *k = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
            fprintf(out, "K,%llu,%llu,%u,%u,%u,%dx%dx%d,%dx%dx%d,%s\n",
                    (unsigned long long)k->start, (unsigned long long)k->end, k->streamId,
                    k->graphId, k->correlationId, k->gridX, k->gridY, k->gridZ, k->blockX,
                    k->blockY, k->blockZ, k->name);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto *m = reinterpret_cast<CUpti_ActivityMemcpy5 *>(record);
            fprintf(out, "M,%llu,%llu,%u,%u,%u,%u,%llu\n", (unsigned long long)m->start,
                    (unsigned long long)m->end, m->streamId, m->graphId, m->correlationId,
                    (unsigned)m->copyKind, (unsigned long long)m->bytes);
        }
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped);
    if (dropped) fprintf(stderr, "kernel_trace: dropped %zu records\n", dropped);
    free(buffer);
}

static void flush_all() {
    cuptiActivityFlushAll(1);
    if (out) fclose(out);
    out = nullptr;
}

extern "C" int InitializeInjection(void) {
    const char *path = getenv("KTRACE_OUT");
    out = fopen(path ? path : "ktrace.csv", "w");
    if (!out) return 0;
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    atexit(flush_all);
    return 1;
}
