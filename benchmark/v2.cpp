// libfive
#include <libfive/tree/tree.hpp>
#include <libfive/tree/archive.hpp>
#include <libfive/render/discrete/heightmap.hpp>

#include "v2.hpp"

int main(int, char**)
{
    auto X = libfive::Tree::X();
    auto Y = libfive::Tree::Y();
    auto t = sqrt((X + 1)*(X + 1) + (Y + 1)*(Y + 1)) - 1.8;

    auto blob = build_v2_blob(t, 256);
    render_v2_blob(blob, Eigen::Matrix4f::Identity());
    free_v2_blob(blob);
}
