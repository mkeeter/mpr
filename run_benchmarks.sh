#!/bin/bash
set -e

# Make sure that the build is fresh
ninja > /dev/null

# Nuke all images
rm -rf *.png

# Print useful information
echo -n "Running benchmarks "
date
whoami
uname
hostname
echo "------------------------------------------------------------"

git rev-parse HEAD
git --no-pager diff | cat

echo "============================================================"
echo "                      2D benchmarks                         "
echo "============================================================"
echo "Text benchmark"
./benchmark/render_2d_table ../benchmark/files/prospero.frep
mkdir -p prospero
mv *.png prospero

echo "------------------------------------------------------------"
echo "Gears (2D)"
./benchmark/render_2d_table ../benchmark/files/involute_gear_2d.frep
mkdir -p gears_2d
mv *.png gears_2d

echo "============================================================"
echo "                      3D benchmarks                         "
echo "============================================================"
echo "Architecture model"
./benchmark/render_3d_table ../benchmark/files/architecture.frep
mkdir -p architecture
mv *.png architecture

echo "------------------------------------------------------------"
echo "Gears (3D)"
./benchmark/render_3d_table ../benchmark/files/involute_gear_3d.frep
mkdir -p gears_3d
mv *.png gears_3d

echo "------------------------------------------------------------"
echo "Bear sculpt"
./benchmark/render_3d_table ../benchmark/files/bear.frep
mkdir -p bear
mv *.png bear
