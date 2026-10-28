// ECS control-plane policy core.
//
// This file intentionally contains no model math and no CUDA.  It is the
// CPU-side bookkeeping layer: heat tracking, tier placement and global LRU
// victim selection.  The Python controller owns tensor storage; this shared
// object only answers placement/eviction policy questions.

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define ECS_EXPORT __declspec(dllexport)
#else
#define ECS_EXPORT __attribute__((visibility("default")))
#endif

struct ECS_EXPORT EcsEntry {
    uint64_t key;
    int32_t tier;
    int32_t device;
    uint64_t bytes;
    double heat;
    uint64_t last_tick;
    uint8_t pinned;
};

static std::mutex g_mutex;
static std::unordered_map<uint64_t, EcsEntry> g_entries;

static double ecs_score(const EcsEntry& entry, uint64_t tick) {
    const double age = tick >= entry.last_tick
        ? double(tick - entry.last_tick)
        : 0.0;
    // Lower score is evicted first.  Heat dominates, age breaks ties.
    return entry.heat - age * 1.0e-9;
}

extern "C" ECS_EXPORT void ecs_reset() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_entries.clear();
}

extern "C" ECS_EXPORT int ecs_touch(
    uint64_t key,
    int32_t tier,
    int32_t device,
    uint64_t bytes,
    uint64_t tick,
    double heat_delta)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_entries.find(key);
    if (it == g_entries.end()) {
        if (g_entries.size() >= 65536) {
            return -2;
        }
        EcsEntry entry{key, tier, device, bytes, heat_delta, tick, 0};
        g_entries.emplace(key, entry);
        return 0;
    }
    EcsEntry& entry = it->second;
    entry.tier = tier;
    entry.device = device;
    entry.bytes = bytes;
    entry.heat = std::max(0.0, entry.heat + heat_delta);
    entry.last_tick = tick;
    return 0;
}

extern "C" ECS_EXPORT int ecs_set_tier(
    uint64_t key, int32_t tier, int32_t device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_entries.find(key);
    if (it == g_entries.end()) return -1;
    it->second.tier = tier;
    it->second.device = device;
    return 0;
}

extern "C" ECS_EXPORT int ecs_remove(uint64_t key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_entries.erase(key) > 0 ? 0 : -1;
}

extern "C" ECS_EXPORT int ecs_choose_victim(
    int32_t tier,
    int32_t device,
    uint64_t required_bytes,
    uint64_t tick,
    uint64_t* out_key)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    const EcsEntry* victim = nullptr;
    double victim_score = 0.0;
    uint64_t victim_age = 0;
    for (const auto& pair : g_entries) {
        const EcsEntry& entry = pair.second;
        if (entry.pinned) continue;
        if (tier >= 0 && entry.tier != tier) continue;
        if (device >= 0 && entry.device != device) continue;
        if (required_bytes > 0 && entry.bytes > required_bytes) {
            // A single entry larger than the request is still evictable when
            // the controller needs to reclaim space; do not skip it.
        }
        const double score = ecs_score(entry, tick);
        const uint64_t age = tick >= entry.last_tick
            ? tick - entry.last_tick
            : 0;
        if (victim == nullptr || score < victim_score ||
            (score == victim_score && age > victim_age)) {
            victim = &entry;
            victim_score = score;
            victim_age = age;
        }
    }
    if (victim == nullptr) return -1;
    if (out_key != nullptr) *out_key = victim->key;
    return 0;
}

extern "C" ECS_EXPORT int ecs_lookup(
    uint64_t key,
    int32_t* out_tier,
    int32_t* out_device,
    uint64_t* out_bytes,
    double* out_heat,
    uint64_t* out_last_tick)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_entries.find(key);
    if (it == g_entries.end()) return -1;
    if (out_tier) *out_tier = it->second.tier;
    if (out_device) *out_device = it->second.device;
    if (out_bytes) *out_bytes = it->second.bytes;
    if (out_heat) *out_heat = it->second.heat;
    if (out_last_tick) *out_last_tick = it->second.last_tick;
    return 0;
}

extern "C" ECS_EXPORT int ecs_entry_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return static_cast<int>(g_entries.size());
}

extern "C" ECS_EXPORT int ecs_plan_chunks(
    const int64_t* expert_ids,
    const int32_t* counts,
    int32_t count,
    int32_t max_experts,
    int32_t max_tokens,
    int32_t* out_starts,
    int32_t* out_lengths,
    int32_t out_capacity)
{
    if (expert_ids == nullptr || counts == nullptr || count < 0) return -1;
    if (max_experts < 1 || max_tokens < 1 || out_capacity < 1) return -1;
    int32_t chunks = 0;
    int32_t start = 0;
    while (start < count && chunks < out_capacity) {
        int32_t end = start;
        int64_t tokens = 0;
        while (end < count && end - start < max_experts) {
            const int32_t next = counts[end] > 0 ? counts[end] : 0;
            if (end > start && tokens + next > max_tokens) break;
            tokens += next;
            ++end;
        }
        if (end == start) {
            ++end;
        }
        out_starts[chunks] = start;
        out_lengths[chunks] = end - start;
        ++chunks;
        start = end;
    }
    return chunks;
}
