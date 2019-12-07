// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#define protected public
#include "renderable.hpp"

// Not actually a benchmark, used to generate a figure for the paper
int main(int, char**)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto t = sqrt((X + 1)*(X + 1) + (Y + 1)*(Y + 1)) - 1.8;

    auto r_ = Renderable::build(t, 128, 2);
    auto r = dynamic_cast<Renderable2D*>(r_.get());
    r->run({Eigen::Matrix4f::Identity()});

    // Save the image using libfive::Heightmap
    libfive::Heightmap out(r->image.size_px, r->image.size_px);
    for (unsigned x=0; x < r->image.size_px; ++x) {
        for (unsigned y=0; y < r->image.size_px; ++y) {
            out.depth(y, x) = r->heightAt(x, y);
        }
    }

    unsigned index=0;
    out.norm = 0xFF000055;
    for (unsigned x=0; x < 2; ++x) {
        for (unsigned y=0; y < 2; ++y) {
            if (r->filled_tiles.filled(index++)) {
                for (unsigned i=x*64; i < (x+1)*64; ++i) {
                    for (unsigned j=y*64; j < (y+1)*64; ++j) {
                        out.norm(j,i) = 0xFFFFFFFF;
                    }
                }
            }
        }
    }
    out.saveNormalPNG("circle2.png");
    index = 0;
    for (unsigned x=0; x < 16; ++x) {
        for (unsigned y=0; y < 16; ++y) {
            if (r->filled_subtiles.filled(index++)) {
                for (unsigned i=x*8; i < (x+1)*8; ++i) {
                    for (unsigned j=y*8; j < (y+1)*8; ++j) {
                        out.norm(j,i) = 0xFFAAAAAA;
                    }
                }
            }
        }
    }
    for (unsigned x=0; x < 16; ++x) {
        for (unsigned y=0; y < 16; ++y) {
            if (r->filled_subtiles.filled(index++)) {
                for (unsigned i=x*8; i < (x+1)*8; ++i) {
                    for (unsigned j=y*8; j < (y+1)*8; ++j) {
                        out.norm(j,i) = 0xFFAAAAAA;
                    }
                }
            }
        }
    }
    out.saveNormalPNG("circle2.png");

    for (unsigned i=0; i < r->queue_pong.count; ++i) {
        const uint32_t subtile = r->queue_pong.data[i];
        const uint32_t subtile_pos = r->pixel_renderer.subtiles.pos(subtile);
        const uint32_t x = subtile_pos % (128/8);
        const uint32_t y = subtile_pos / (128/8);
        for (unsigned i=x*8; i < (x+1)*8; ++i) {
            for (unsigned j=y*8; j < (y+1)*8; ++j) {
                if (out.depth(j,i)) {
                    out.norm(j,i) = 0xFFAA6666;
                } else {
                    out.norm(j,i) = 0xFF000000;
                }
            }
        }
    }
    out.saveNormalPNG("circle3.png");
}
