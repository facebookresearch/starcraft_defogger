/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#include "coarse_conv_featurizer_stats.h"
#include "constants.h"

template<typename T> using vector = std::vector<T>;
using Unit = torchcraft::replayer::Unit;
using Frame = torchcraft::replayer::Frame;
using Order = torchcraft::replayer::Order;
template<typename T> using Batch = vector<Frame>;
using Feature = vector<vector<vector<double>>>;


CoarseConvFeaturizerStats::CoarseConvFeaturizerStats(
    size_t resX, size_t resY, int32_t strideX, int32_t strideY,
    size_t from, size_t until
    ) : CoarseConvFeaturizer(resX, resY, strideX, strideY, from, until) {
  size_t size = 0;
#define SET_INDS(NAME, SIZE) NAME = size; NAME##_size = SIZE; size += SIZE;
  CoarseConvFeaturizerStats_DO_FEATURES(SET_INDS);
#undef SET_INDS
  this->feature_size = size;
}

void CoarseConvFeaturizerStats::featurize_unit(
    Feature &feats, Unit &u, bool perspective, bool full) const {
  auto offset = (perspective == u.playerId) ? 0 : feature_size;
  auto visible = (u.visible >> perspective) & 0x1;
  if (!full && !visible) return; // Don't featurize if we can't see unit

  int32_t nBinX = feats.size();
  int32_t nBinY = feats[0].size();

  auto order = u.orders.back();
  auto tX = order.targetX == -1 ? u.x : order.targetX;
  auto tY = order.targetY == -1 ? u.y : order.targetY;
  auto dX = tX - u.x;
  auto dY = tY - u.y;

  int32_t binX = std::min(u.x / (int) strideX, nBinX - 1);
  int32_t binY = std::min(u.y / (int) strideY, nBinY - 1);
  for (auto i = 0U; i < resX; i++) {
    for (auto j = 0U; j < resY; j++) {
      int32_t x = binX - i;
      int32_t y = binY - j;
      if (x < 0 || y < 0) continue;
      if (!((strideX * x < u.x) && (u.x < strideX * x + (int32_t)resX))) continue;
      if (!((strideY * y < u.y) && (u.y < strideY * y + (int32_t)resY))) continue;
      auto &feat = feats[x][y];
      feat[offset + N_UNITS]    ++;
      feat[offset + VELOCITY_X] += u.velocityX;
      feat[offset + VELOCITY_Y] += u.velocityY;
      feat[offset + HEALTH]     += u.health / div_hp;
      feat[offset + SHIELD]     += u.shield / div_shield;
      feat[offset + ARMOR]      += u.armor;
      feat[offset + TARGET_X]   += dX / div_coord;
      feat[offset + TARGET_Y]   += dY / div_coord;
      feat[offset + TARGET_DIST]+= sqrt(dX * dX + dY * dY);
      feat[offset + GWATTACK]   += u.groundATK / div_dmg;
      feat[offset + GWDMGTYPE + u.groundDmgType] += 1;
      feat[offset + GWRANGE]    += u.groundRange / div_coord;
      feat[offset + AWATTACK]   += u.airATK / div_dmg;
      feat[offset + AWDMGTYPE + u.airDmgType] += 1;
      feat[offset + AWRANGE]    += u.airRange / div_coord;
      feat[offset + N_FLYING]   += (u.flags & Unit::Flags::Flying) > 0 ? 1 : 0;
      feat[offset + N_BUILDING] += (torchcraft::BW::data::IsBuilding[u.type]) ? 1 : 0;
    }
  }
}
