/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#include "coarse_conv_featurizer_units.h"
#include "constants.h"

namespace tc = torchcraft;

template<typename T> using vector = std::vector<T>;
using Unit = tc::replayer::Unit;
using Frame = tc::replayer::Frame;
using Order = tc::replayer::Order;
template<typename T> using Batch = vector<Frame>;
using Feature = vector<vector<vector<double>>>;

CoarseConvFeaturizerUnitTypes::CoarseConvFeaturizerUnitTypes(
    size_t resX, size_t resY, int32_t strideX, int32_t strideY,
    size_t from, size_t until
    ) : CoarseConvFeaturizer(resX, resY, strideX, strideY, from, until) {

    this->feature_size = 118;
    typemapper.fill(117);
    size_t i = 0;
    for (auto t : tc::BW::UnitType::_values()) {
      typemapper.at(t._to_integral()) = i;
      i++;
    }
  }

void CoarseConvFeaturizerUnitTypes::featurize_unit(
    Feature &feats, Unit &u, bool perspective, bool full) const {
  auto offset = (perspective == u.playerId) ? 0 : feature_size;
  auto visible = (u.visible >> perspective) & 0x1;
  if (!full && !visible) return; // Don't featurize if we can't see unit
  inc_feature(feats, offset + typemapper.at(u.type), u.x, u.y);
}

void CoarseConvFeaturizerUnitTypes::reduce_frame(std::ostream &out,
    Frame* frame) const {
  ffbs::ReducedUnitTypesT ut;
  ut.version = rfVersion;
  for (auto& unit : frame->units[0]) {
    ut.units_p0.emplace_back(unit.id, unit.x, unit.y, unit.type, unit.visible);
  }
  for (auto& unit : frame->units[1]) {
    ut.units_p1.emplace_back(unit.id, unit.x, unit.y, unit.type, unit.visible);
  }
  for (auto& unit : frame->units[-1]) {
    ut.units_n.emplace_back(unit.id, unit.x, unit.y, unit.type, unit.visible);
  }

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(ffbs::ReducedUnitTypes::Pack(fbb, &ut));
  size_t size = fbb.GetSize();
  writePOD(out, size);
  out.write(reinterpret_cast<char const*>(fbb.GetBufferPointer()), fbb.GetSize());
}

CoarseConvFeaturizer::Reduced* CoarseConvFeaturizerUnitTypes::read_reduced_frame(
    std::istream& in) const {
  size_t size;
  readPOD(size, in);

  ReducedUnitTypes* red = new ReducedUnitTypes();
  red->data.resize(size);
  in.read(red->data.data(), size);
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t*>(red->data.data()), red->data.size());
  if (!ffbs::VerifyReducedUnitTypesBuffer(verifier)) {
    throw std::runtime_error("corrupted data");
  }
  red->d = ffbs::GetReducedUnitTypes(red->data.data());

  return red;
}

CoarseConvFeaturizer::Reduced* CoarseConvFeaturizerUnitTypes::combine_reduced_frames(
    std::vector<CoarseConvFeaturizer::Reduced*>::const_iterator begin,
    std::vector<CoarseConvFeaturizer::Reduced*>::const_iterator end,
    int32_t perspective, bool full) const {
  // This is currently modelled after CoarseConvFeaturizer::combine
  ReducedUnitTypes* combined = new ReducedUnitTypes();
  for (auto rframe = begin; rframe != end; ++rframe) {
    auto next_frame = reinterpret_cast<ReducedUnitTypes*>(*rframe);
    if (next_frame == nullptr) {
      throw std::runtime_error("Mismatched frame data");
    }
    // For units, accumulate presence
    for (auto player_id : {0, 1, -1}) {
      flatbuffers::Vector<ffbs::UnitType const*> const* player_units = nullptr;
      if (player_id == 0) {
        player_units = next_frame->d->units_p0();
      } else if (player_id == 1) {
        player_units = next_frame->d->units_p1();
      } else if (player_id == -1) {
        player_units = next_frame->d->units_n();
      }

      auto& combined_units = combined->units[player_id];

      // Build dictionary of uid -> position in current frame unit vector
      std::unordered_map<int32_t, int32_t> next_idx;
      if (player_units != nullptr) {
        for (size_t i = 0; i < player_units->size(); i++) {
          next_idx[player_units->Get(i)->id()] = i;
        }
      }

      for (auto unit = combined_units.begin(); unit != combined_units.end(); ) {
        // If unit isn't in next frame, it must have died, so we delete it.
        // This doesn't delete units that went into the FOW, although it will
        // delete garrisoned marines I think.
        if (next_idx.count(unit->id) == 0) {
          unit = combined_units.erase(unit);
        } else {
          ++unit;
        }
      }

      std::unordered_map<int32_t, int32_t> combined_idx;
      for (size_t i = 0; i < combined_units.size(); i++) {
        combined_idx[combined_units[i].id] = i;
      }

      // Iterate over units in next frame
      if (player_units) {
        for (const auto& unit : *player_units) {
          auto visible = (unit->visible() >> perspective) & 0x1;
          if (!full && !visible) continue; // Don't featurize if we can't see unit

          if (combined_idx.count(unit->id()) == 0) {
            // Unit wasn't in current frame, add it
            combined_units.push_back({unit->id(),
                unit->x(), unit->y(), unit->type(), unit->visible()});
          } else {
            int32_t i = combined_idx[unit->id()];
            combined_units[i] = {unit->id(),
                unit->x(), unit->y(), unit->type(), unit->visible()};
          }
        }
      }
    }
  }

  return combined;
}

void CoarseConvFeaturizerUnitTypes::featurize_reduced_frame(
    Feature& feature,
    Reduced* rframe,
    int32_t perspective,
    bool full) const {
  auto urframe = reinterpret_cast<ReducedUnitTypes*>(rframe);
  assert(urframe != nullptr);

  for (auto player : {perspective, 1-perspective}) {
    auto offset = (perspective == player) ? 0 : feature_size;
    for (auto& u : urframe->units[player]) {
      auto visible = (u.visible >> perspective) & 0x1;
      if (!full && !visible) {
        // Don't featurize if we can't see unit
        continue;
      }
      inc_feature(feature, offset + typemapper.at(u.type), u.x, u.y);
    }
  }
}

void CoarseConvFeaturizerUnitTypes::featurize_visibility(
    Feature& feature,
    Reduced* rframe,
    int32_t perspective) const {
  auto urframe = reinterpret_cast<ReducedUnitTypes*>(rframe);
  assert(urframe != nullptr);

  for (auto& u : urframe->units[perspective]) {
    auto range = tc::BW::data::SightRange[u.type] / tc::BW::XYPixelsPerWalktile;

    // Mark all points in a circle around the unit's position.
    // Use https://en.wikipedia.org/wiki/Midpoint_circle_algorithm for
    // rasterization.
    int32_t x = range;
    int32_t y = 0;
    int32_t err = 0;
    while (x >= y) {
      mark_feature_vline(feature, 0, u.x - y, u.y - x, u.y + x);
      mark_feature_vline(feature, 0, u.x + y, u.y - x, u.y + x);
      mark_feature_vline(feature, 0, u.x - x, u.y - y, u.y + y);
      mark_feature_vline(feature, 0, u.x + x, u.y - y, u.y + y);

      y++;
      if (err <= 0) {
        err += 2 * y + 1;
      }
      if (err > 0) {
        x--;
        err -= 2 * x + 1;
      }
    }
  }
}

void CoarseConvFeaturizerUnitTypes::featurize_invfow(
    Feature& feature,
    Reduced* rframe,
    int32_t perspective,
    TilesInfo& tinfo,
    FogOfWar& fow) const {
  auto urframe = reinterpret_cast<ReducedUnitTypes*>(rframe);
  assert(urframe != nullptr);

  for (auto& u : urframe->units[perspective]) {
    auto range = tc::BW::data::SightRange[u.type] / tc::BW::XYPixelsPerWalktile;
    auto isFlyer = tc::BW::data::IsFlyer[u.type];
    // XXX Cannot account for lifted units right now since unit flags are not
    // part of the reduced frame data.
    fow.revealSightAt(tinfo, u.x, u.y, range, isFlyer,
        0 /* Don't care about frame numbers */);
  }

  // Pool into feature bins. Every bin for which we have at least 50% build tile
  // visibility will be marked as visible
  assert(resX % 4 == 0);
  assert(resY % 4 == 0);
  assert(strideX % 4 == 0);
  assert(strideY % 4 == 0);
  int32_t nBinY = feature.size();
  int32_t nBinX = feature[0].size();
  int32_t startTileX = 0;
  for (int32_t x = 0; x < nBinX; x++) {
    int32_t endTileX = std::min(tinfo.mapTileWidth, unsigned(startTileX + resX/4));
    int32_t startTileY = 0;
    for (int32_t y = 0; y < nBinY; y++) {
      int32_t endTileY = std::min(tinfo.mapTileHeight, unsigned(startTileY + resY/4));

      int32_t countvis = 0;
      for (int32_t tx = startTileX; tx < endTileX; tx++) {
        for (int32_t ty = startTileY; ty < endTileY; ty++) {
          auto& tt = tinfo.tiles[TilesInfo::tilesWidth * ty + tx];
          if (tt.visible) {
            countvis++;
          } else {
            countvis--;
          }
        }
      }
      if (countvis > 0) {
        feature[y][x][0] = 1;
      }

      startTileY += strideY/4;
    }
    startTileX += strideX/4;
  }
}

void CoarseConvFeaturizerUnitTypes::inc_feature(Feature& feature,
    int32_t c, int32_t x, int32_t y) const {
  int32_t nBinY = feature.size();
  int32_t nBinX = feature[0].size();

  // Determine resulting bins for this position.
  // The last kernel applications that contains it will be placed at
  // (floor(x/strideX), floor(y/strideY). The number of kernels
  // applications containing it (e.g. on the X axis) is given by
  // ceil((resX - x % strideX) / strideX). Here, (resX - x % strideX) is the
  // offset of x within the first kernel application (which happens at a
  // multiple of strideX by definition).
  // Note that if stride > res, the position might not end up in any
  // application.
  int32_t maxbX = std::min(x / strideX, nBinX - 1) + 1;
  int32_t maxbY = std::min(y / strideY, nBinY - 1) + 1;
  int32_t minbX = std::max(0,
      maxbX - (int32_t(resX) - (x % strideX) + strideX - 1) / strideX);
  int32_t minbY = std::max(0,
      maxbY - (int32_t(resY) - (y % strideY) + strideY - 1) / strideY);

  for (int32_t by = minbY; by < maxbY; by++) {
    for (int32_t bx = minbX; bx < maxbX; bx++) {
      ++feature[by][bx][c];
    }
  }
}

void CoarseConvFeaturizerUnitTypes::mark_feature_vline(Feature& feature,
    int32_t c, int32_t x, int32_t y1, int32_t y2) const {
  assert(y2 >= y1);
  int32_t nBinY = feature.size();
  int32_t nBinX = feature[0].size();

  // See inc_feature() for an explanation regarding translation of positions. As
  // an optimization we're going to mark lines in feature (binned) space instead
  // of walk tile space, which is ok since we're not interested in absolute
  // counts but rather in binary "marking".
  int32_t maxbX = std::min(x / strideX, nBinX - 1) + 1;
  int32_t maxbY1 = std::min(y1 / strideY, nBinY - 1) + 1;
  int32_t maxbY2 = std::min(y2 / strideY, nBinY - 1) + 1;
  int32_t minbX = std::max(0,
      maxbX - (int32_t(resX) - (x % strideX) + strideX - 1) / strideX);
  int32_t minbY1 = std::max(0,
      maxbY1 - (int32_t(resY) - (y1 % strideY) + strideY - 1) / strideY);

  for (int32_t by = minbY1; by < maxbY2; by++) {
    for (int32_t bx = minbX; bx < maxbX; bx++) {
      feature[by][bx][c] = 1.0;
    }
  }
}
