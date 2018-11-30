/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Tile {
  int x = 0;
  int y = 0;
  int height = 0;
  int lastSeen = 0;
  bool visible = false;
};

struct TilesInfo {
  TilesInfo(py::array_t<uint8_t> ground_height);

  static const unsigned tilesWidth = 256;
  static const unsigned tilesHeight = 256;

  std::vector<Tile> tiles;
  unsigned mapTileWidth = 0;
  unsigned mapTileHeight = 0;

  Tile* tryGetTile(int x, int y);
  void clearVisible();
};

// This class is based on OpenBW
class FogOfWar {
  struct sight_values_t {
    struct maskdat_node_t {
      size_t prev;
      size_t prev2;
      int relative_tile_index;
      int x;
      int y;
    };
    int max_width, max_height;
    int min_width, min_height;
    int min_mask_size;
    int ext_masked_count;
    std::vector<maskdat_node_t> maskdat;
  };

  std::array<sight_values_t, 12> sight_values;

  void generateSightValues();

 public:
  FogOfWar();
  void revealSightAt(
      TilesInfo& tt,
      int x,
      int y,
      int range,
      bool in_air,
      int currentFrame) const;
};
