/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#include "coarse_conv_featurizer.h"
#include "constants.h"
#include "replayer/zstdstream.h"

template<typename T> using vector = std::vector<T>;
using Unit = torchcraft::replayer::Unit;
using Frame = torchcraft::replayer::Frame;
using Order = torchcraft::replayer::Order;
template<typename T> using Batch = vector<Frame>;
using Feature = vector<vector<vector<double>>>;

Frame combine(const vector<Frame*> &frames, int32_t perspective, bool full) {
  Frame combined;
  for (const auto *next_frame : frames) {
    // For units, accumulate presence and commands
    for (const auto& player : next_frame->units) {
      auto& player_id = player.first;
      auto& player_units = player.second;
      auto& combined_units = combined.units[player_id];

      // Build dictionary of uid -> position in current frame unit vector
      std::unordered_map<int32_t, int32_t> next_idx;
      for (size_t i = 0; i < player_units.size(); i++)
        next_idx[player_units[i].id] = i;

      for (auto unit = combined_units.begin();
          unit != combined_units.end(); ) {
        // If unit isn't in next frame, it must have died, so we delete it.
        // This doesn't delete units that went into the FOW, although it will
        // delete garrisoned marines I think.
        if (next_idx.count(unit->id) == 0)
          unit = combined_units.erase(unit);
        else unit++;
      }

      std::unordered_map<int32_t, int32_t> combined_idx;
      for (size_t i = 0; i < combined_units.size(); i++)
        combined_idx[combined_units[i].id] = i;

      // Iterate over units in next frame
      for (const auto& unit : player_units) {
        auto visible = (unit.visible >> perspective) & 0x1;
        if (!full && !visible) continue; // Don't featurize if we can't see unit

        if (combined_idx.count(unit.id) == 0) {
          // Unit wasn't in current frame, add it
          combined_units.push_back(unit);
        } else {
          int32_t i = combined_idx[unit.id];
          // Take unit state from next frame but accumulate orders
          // so as to have a vector of all the orders taken
          std::vector<Order> ords = std::move(combined_units[i].orders);
          ords.reserve(ords.size() + unit.orders.size());
          for (auto& ord : unit.orders) {
            if (ords.empty() || !(ord == ords.back())) {
              ords.push_back(ord);
            }
          }
          combined_units[i] = unit;
          combined_units[i].orders = std::move(ords);
        }
      }
      // For resources: keep the ones of the next frame
      if (next_frame->resources.find(player_id) != next_frame->resources.end()) {
        auto next_res = next_frame->resources.at(player_id);
        combined.resources[player_id].ore = next_res.ore;
        combined.resources[player_id].gas = next_res.gas;
        combined.resources[player_id].used_psi = next_res.used_psi;
        combined.resources[player_id].total_psi = next_res.total_psi;
      }
    }
    // For other stuff, simply keep that of next_frame
    combined.actions = next_frame->actions;
    combined.bullets = next_frame->bullets;
    combined.reward = next_frame->reward;
    combined.is_terminal = next_frame->is_terminal;
  }

  return combined;
}

vector<Feature> CoarseConvFeaturizer::featurize(
    const vector<vector<Frame*>> &batch,
    const vector<std::pair<size_t, size_t>> map_sizes,
    int32_t perspective,
    bool full) const {
  // Featurize first time step of each batch
  assert(perspective == 0 || perspective == 1);
  vector<Feature> features;
  for (size_t i = 0; i < batch.size(); i++) {
    auto nBinX = (double)(map_sizes[i].second - resX) / strideX + 1;
    auto nBinY = (double)(map_sizes[i].first - resY) / strideY + 1;
    if (nBinX != (int)nBinX)
      std::cerr << "WARNING: X dimension of " << map_sizes[i].second <<
        " is not evenly tiled by kW " << resX <<
        " and stride " << strideX << " because you get " << nBinX << "bins\n";
    if (nBinY != (int)nBinY)
      std::cerr << "WARNING: Y dimension of " << map_sizes[i].first <<
        " is not evenly tiled by kW " << resY <<
        " and stride " << strideY << " because you get " << nBinY << "bins\n";
    Feature feat(ceil(nBinY), vector<vector<double>>(ceil(nBinX),
          vector<double>(2 * feature_size, 0.0)));

    auto frame = combine(batch[i], perspective, full);

    for (auto unit : frame.units[perspective])
      featurize_unit(feat, unit, perspective, full);
    for (auto unit : frame.units[1-perspective])
      featurize_unit(feat, unit, perspective, full);

    features.push_back(std::move(feat));
  }

  return features;
}

vector<char> CoarseConvFeaturizer::reduce(vector<Frame*> const& frames) const {
  std::ostringstream oss;
  zstd::ostreambuf zbuf(oss.rdbuf());
  std::ostream os(&zbuf);
  writePOD(os, frames.size());
  for (auto const& frame : frames) {
    reduce_frame(os, frame);
  }
  os.flush();
  oss.flush();
  auto str = oss.str();
  // XXX all these copies...
  vector<char> data(str.begin(), str.end());
  return data;
}

struct membuf : std::streambuf {
 public:
  explicit membuf(vector<char> const& data) {
    char* p(const_cast<char*>(data.data()));
    setg(p, p, p + data.size());
  }
};

vector<Feature> CoarseConvFeaturizer::featurize_reduced(
    vector<char> const& data,
    size_t skip_frames,
    size_t combine_frames,
    std::pair<size_t, size_t> const &map_size,
    int32_t perspective,
    bool full) const {
  assert(perspective == 0 || perspective == 1);

  // Map size is (ydim, xdim)
  auto nBinX = (double)(map_size.second - resX) / strideX + 1;
  auto nBinY = (double)(map_size.first - resY) / strideY + 1;
  if (nBinX != (int)nBinX)
    std::cerr << "WARNING: X dimension of " << map_size.second <<
      " is not evenly tiled by kW " << resX <<
      " and stride " << strideX << " because you get " << nBinX << "bins\n";
  if (nBinY != (int)nBinY)
    std::cerr << "WARNING: Y dimension of " << map_size.first <<
      " is not evenly tiled by kW " << resY <<
      " and stride " << strideY << " because you get " << nBinY << "bins\n";

  vector<Feature> features;
  membuf mbuf(data);
  zstd::istreambuf zbuf(&mbuf);
  std::istream is(&zbuf);

  size_t nframes;
  readPOD(nframes, is);

  // Read all frames
  std::vector<Reduced*> rframes;
  for (auto i = 0U; i < nframes; i++) {
    auto rf = read_reduced_frame(is);
    if (this->until == 0 || i >= this->from) {
      rframes.push_back(rf);
    } else if (i >= this->until) {
      break;
    }
  }

  auto it = rframes.begin();
  while (it != rframes.end()) {
    auto begin = it;
    auto end = std::min(it + combine_frames, rframes.end());
    auto rframe = combine_reduced_frames(begin, end, perspective, full);
    it = std::min(it + skip_frames, rframes.end());

    // Featurize
    features.emplace_back(ceil(nBinY), vector<vector<double>>(ceil(nBinX),
          vector<double>(2 * feature_size, 0.0)));
    featurize_reduced_frame(features.back(), rframe, perspective, full);

    delete rframe;
  }

  // Clean up
  for (size_t j = 0; j < rframes.size(); j++) {
    delete rframes[j];
  }
  return features;
}

// featurizer for (perspective=pers, full=False) and (perspective=pers, full=True)
// Also returns a visibility channel as a separate feature
vector<vector<Feature>> CoarseConvFeaturizer::featurize_reduced_all(
    vector<char> const& data,
    size_t skip_frames,
    size_t combine_frames,
    std::pair<size_t, size_t> const &map_size,
    int32_t perspective,
    bool visibility,
    py::array_t<uint8_t> ground_height) const {
  assert(perspective == 0 || perspective == 1);

  auto nBinX = (double)(map_size.second - resX) / strideX + 1;
  auto nBinY = (double)(map_size.first - resY) / strideY + 1;
  if (nBinX != (int)nBinX)
    std::cerr << "WARNING: X dimension of " << map_size.second <<
      " is not evenly tiled by kW " << resX <<
      " and stride " << strideX << " because you get " << nBinX << "bins\n";
  if (nBinY != (int)nBinY)
    std::cerr << "WARNING: Y dimension of " << map_size.first <<
      " is not evenly tiled by kW " << resY <<
      " and stride " << strideY << " because you get " << nBinY << "bins\n";

  vector<Feature> features1;
  vector<Feature> features2;
  vector<Feature> featuresV;
  membuf mbuf(data);
  zstd::istreambuf zbuf(&mbuf);
  std::istream is(&zbuf);

  size_t nframes;
  readPOD(nframes, is);

  // Read all frames
  std::vector<Reduced*> rframes;
  for (auto i = 0U; i < nframes; i++) {
    auto rf = read_reduced_frame(is);
    if (this->until == 0 || i >= this->from) {
      rframes.push_back(rf);
    } else if (i >= this->until) {
      break;
    } else {
      delete rf;
    }
  }

  // Prepare tiles info and fog of war
  TilesInfo tinfo(ground_height);
  FogOfWar fow;

  auto it = rframes.begin();
  while (rframes.size() != 0 && it != rframes.end()) {
    auto begin = it;
    auto end = std::min(it + combine_frames, rframes.end());
    auto rframe1 = combine_reduced_frames(begin, end, perspective, false);
    auto rframe2 = combine_reduced_frames(begin, end, perspective, true);
    it = std::min(it + skip_frames, rframes.end());

    // Featurize
    features1.emplace_back(ceil(nBinY), vector<vector<double>>(ceil(nBinX),
          vector<double>(2 * feature_size, 0.0)));
    features2.emplace_back(ceil(nBinY), vector<vector<double>>(ceil(nBinX),
          vector<double>(2 * feature_size, 0.0)));
    featurize_reduced_frame(features1.back(), rframe1, perspective, false);
    featurize_reduced_frame(features2.back(), rframe2, perspective, true);
    if (visibility) {
      featuresV.emplace_back(ceil(nBinY), vector<vector<double>>(ceil(nBinX),
            vector<double>(1, 0.0)));

      tinfo.clearVisible();
      featurize_invfow(featuresV.back(), rframe1, perspective, tinfo, fow);
//      featurize_visibility(featuresV.back(), rframe1, perspective);
    }

    delete rframe1;
    delete rframe2;
  }

  // Clean up
  for (size_t j = 0; j < rframes.size(); j++) {
    delete rframes[j];
  }
  vector<vector<Feature>> ret;
  ret.emplace_back(std::move(features1));
  ret.emplace_back(std::move(features2));
  if (visibility) {
    ret.emplace_back(std::move(featuresV));
  }
  return ret;
}
