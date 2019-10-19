#include "check.hpp"
#include "renderable.hpp"
#include "gpu_interval.hpp"

Renderable* Renderable::build(libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks, uint32_t num_fill_blocks,
            uint32_t num_subtapes)
{
    auto out = CUDA_MALLOC(Renderable, 1);
    new (out) Renderable(
            image_size_px, tile_size_px,
            num_interval_blocks, num_fill_blocks, num_subtapes);
    return out;
}

Renderable::Renderable(libfive::Tree tree,
            uint32_t image_size_px, uint32_t tile_size_px,
            uint32_t num_interval_blocks, uint32_t num_fill_blocks,
            uint32_t num_subtapes)
    : tape(Tape::build(tree)),

      IMAGE_SIZE_PX(image_size_px),
      TILE_SIZE_PX(tile_size_px),
      TILE_COUNT(IMAGE_SIZE_PX / TILE_SIZE_PX),
      TOTAL_TILES(TILE_COUNT * TILE_COUNT),

      NUM_INTERVAL_BLOCKS(num_interval_blocks),
      THREADS_PER_INTERVAL_BLOCK(TILE_COUNT / NUM_INTERVAL_BLOCKS),

      NUM_FILL_BLOCKS(num_fill_blocks),
      NUM_SUBTAPES(num_subtapes),

      scratch(CUDA_MALLOC(uint8_t,
          std::max(TOTAL_TILES * (sizeof(Interval) * tape.num_regs +
                                  max(1, tape.num_csg_choices)),
                   sizeof(float) * tape.num_regs * NUM_FILL_BLOCKS *
                       TILE_SIZE_PX * TILE_SIZE_PX))),
      regs_i(reinterpret_cast<Interval*>(scratch)),
      csg_choices(scratch + TOTAL_TILES * sizeof(Interval) * tape.num_regs),
      regs_f(reinterpret_cast<float*>(scratch)),

      tiles(CUDA_MALLOC(uint32_t, 2 * TOTAL_TILES)),
      active_tiles(0),
      filled_tiles(0),

      subtapes(CUDA_MALLOC(Subtape, NUM_SUBTAPES)),
      active_subtapes(0)
{
    // lol
}
