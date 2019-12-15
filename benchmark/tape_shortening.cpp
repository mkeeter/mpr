#include <iostream>
#include <fstream>

// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#define protected public
#include "renderable.hpp"

// Not actually a benchmark, used to generate a figure for the paper
int main(int, char**)
{
    std::ifstream ifs;
    const auto filename = "../benchmark/files/prospero_long.frep";
    ifs.open(filename);
    libfive::Tree t(0.0);
    if (ifs.is_open()) {
        auto a = libfive::Archive::deserialize(ifs);
        t = a.shapes.front().tree;
    } else {
        std::cerr << "Could not open " << filename << "\n";
        exit(1);
    }

    const auto size = 1024;
    auto r_ = Renderable::build(t, size, 2);
    auto r = dynamic_cast<Renderable2D*>(r_.get());
    r->run({Eigen::Matrix4f::Identity()});

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(r->image.size_px, r->image.size_px);
    for (unsigned x=0; x < r->image.size_px; ++x) {
        for (unsigned y=0; y < r->image.size_px; ++y) {
            out.depth(y, x) = r->heightAt(x, y);
        }
    }
    std::cout << "Initial clauses: " << r->tape.num_clauses << "\n";
    out.savePNG("hello_world.png");

    for (unsigned i=0; i < r->queue_ping.count; ++i) {
        const uint32_t tile = r->queue_ping.data[i];
        const uint32_t tile_pos = r->subtile_renderer.tiles.pos(tile);
        const uint32_t x = tile_pos % (size/64);
        const uint32_t y = tile_pos / (size/64);

        uint32_t head = r->subtile_renderer.tiles.head(tile);
        uint32_t len = 0;
        while (head) {
            std::cout << r->subtapes.start[head] << "\n";
            len += LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE - r->subtapes.start[head];
            head = r->subtapes.next[head];
        }
        if (len > 0xFFFF) {
            std::cout << "toooo big" << len << "\n";
        }
        const uint32_t pix = 0xFF000000 | ((0xFFFF) & len);

        for (unsigned i=x*64; i < (x+1)*64; ++i) {
            for (unsigned j=y*64; j < (y+1)*64; ++j) {
                out.norm(j,i) = pix;
            }
        }
    }
    out.saveNormalPNG("tile_tape_lengths.png");
    out.norm = 0;

    for (unsigned i=0; i < r->queue_pong.count; ++i) {
        const uint32_t subtile = r->queue_pong.data[i];
        const uint32_t subtile_pos = r->pixel_renderer.subtiles.pos(subtile);
        const uint32_t x = subtile_pos % (size/8);
        const uint32_t y = subtile_pos / (size/8);

        uint32_t head = r->subtile_renderer.subtiles.head(subtile);
        uint32_t len = 0;
        while (head) {
            std::cout << r->subtapes.start[head] << "\n";
            len += LIBFIVE_CUDA_SUBTAPE_CHUNK_SIZE - r->subtapes.start[head];
            head = r->subtapes.next[head];
        }
        if (len > 0xFFFF) {
            std::cout << "toooo big" << len << "\n";
        }
        const uint32_t pix = 0xFF000000 | ((0xFFFF) & len);

        for (unsigned i=x*8; i < (x+1)*8; ++i) {
            for (unsigned j=y*8; j < (y+1)*8; ++j) {
                out.norm(j,i) = pix;
            }
        }
    }
    out.saveNormalPNG("subtile_tape_lengths.png");
}
