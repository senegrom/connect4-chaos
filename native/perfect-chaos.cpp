#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace perfect_chaos {

using Mask = std::uint64_t;

constexpr std::int8_t kLoss = -1;
constexpr std::int8_t kDraw = 0;
constexpr std::int8_t kWin = 1;
constexpr std::int8_t kUnknown = 2;
constexpr std::int8_t kPlaying = 3;

enum class ActionType : std::uint8_t {
  Drop,
  Flip,
  RotateClockwise,
  RotateCounterClockwise,
};

struct Action {
  ActionType type = ActionType::Flip;
  std::uint8_t column = 0;
};

struct State {
  Mask mover = 0;
  Mask opponent = 0;
  std::uint8_t rows = 0;
  std::uint8_t columns = 0;
};

struct Key {
  Mask mover = 0;
  Mask opponent = 0;
  std::uint8_t rows = 0;
  std::uint8_t columns = 0;

  bool operator==(const Key& other) const {
    return mover == other.mover
      && opponent == other.opponent
      && rows == other.rows
      && columns == other.columns;
  }
};

struct KeyHash {
  std::size_t operator()(const Key& key) const {
    std::uint64_t value = key.mover * 0x9e3779b97f4a7c15ULL;
    value ^= key.opponent + 0x517cc1b727220a95ULL;
    value ^= (static_cast<std::uint64_t>(key.rows) << 8U) | key.columns;
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return static_cast<std::size_t>(value);
  }
};

struct CanonicalState {
  State state;
  Key key;
  bool mirrored = false;
};

struct Transition {
  std::int8_t terminal = kPlaying;
  State state;
};

struct Edge {
  Action action;
  std::int8_t terminal = kPlaying;
  std::uint32_t next = std::numeric_limits<std::uint32_t>::max();
};

struct Predecessor {
  std::uint32_t node = 0;
  std::uint8_t edge = 0;
};

struct Node {
  State state;
  std::vector<Edge> edges;
  std::vector<Predecessor> predecessors;
};

struct Graph {
  std::vector<Node> nodes;
  bool root_mirrored = false;
  std::uint8_t root_columns = 0;
};

struct Result {
  std::int8_t value = kDraw;
  std::uint32_t rank = 0;
  std::uint32_t states = 0;
  std::uint32_t wins = 0;
  std::uint32_t draws = 0;
  std::uint32_t losses = 0;
  Action action;
  bool has_action = false;
};

int stride(const State& state) {
  return static_cast<int>(state.rows) + 1;
}

Mask bit(const State& state, int column, int row_from_bottom) {
  return Mask{1} << (column * stride(state) + row_from_bottom);
}

void validate_shape(const State& state) {
  if (state.rows == 0 || state.columns == 0) {
    throw std::runtime_error("Board dimensions must be positive.");
  }
  const int bits = static_cast<int>(state.columns) * (static_cast<int>(state.rows) + 1);
  if (bits > 63) {
    throw std::runtime_error("Native Chaos masks require at most 63 sentinel-layout bits.");
  }
  if ((state.mover & state.opponent) != 0) {
    throw std::runtime_error("Mover and opponent masks overlap.");
  }
}

bool has_win(Mask pieces, int shift_stride, int connect) {
  const std::array<int, 4> shifts = {1, shift_stride, shift_stride - 1, shift_stride + 1};
  for (const int shift : shifts) {
    Mask run = pieces;
    for (int offset = 1; offset < connect; offset += 1) {
      run &= pieces >> (offset * shift);
    }
    if (run != 0) return true;
  }
  return false;
}

bool is_full(const State& state) {
  const Mask occupied = state.mover | state.opponent;
  for (int column = 0; column < state.columns; column += 1) {
    if ((occupied & bit(state, column, state.rows - 1)) == 0) return false;
  }
  return true;
}

State mirror(const State& state) {
  State mirrored{0, 0, state.rows, state.columns};
  for (int column = 0; column < state.columns; column += 1) {
    const int target_column = state.columns - 1 - column;
    for (int row = 0; row < state.rows; row += 1) {
      const Mask source = bit(state, column, row);
      const Mask target = bit(mirrored, target_column, row);
      if ((state.mover & source) != 0) mirrored.mover |= target;
      if ((state.opponent & source) != 0) mirrored.opponent |= target;
    }
  }
  return mirrored;
}

CanonicalState canonicalize(const State& state) {
  validate_shape(state);
  const State reflected = mirror(state);
  const bool use_reflection = reflected.mover < state.mover
    || (reflected.mover == state.mover && reflected.opponent < state.opponent);
  const State selected = use_reflection ? reflected : state;
  return {
    selected,
    {selected.mover, selected.opponent, selected.rows, selected.columns},
    use_reflection,
  };
}

Action reflect_action(const Action& action, int columns) {
  if (action.type == ActionType::Drop) {
    return {ActionType::Drop, static_cast<std::uint8_t>(columns - 1 - action.column)};
  }
  if (action.type == ActionType::RotateClockwise) {
    return {ActionType::RotateCounterClockwise, 0};
  }
  if (action.type == ActionType::RotateCounterClockwise) {
    return {ActionType::RotateClockwise, 0};
  }
  return action;
}

State transform(const State& state, ActionType action) {
  if (action != ActionType::Flip
      && action != ActionType::RotateClockwise
      && action != ActionType::RotateCounterClockwise) {
    throw std::runtime_error("Only transformations can enter the transform function.");
  }
  State transformed{
    0,
    0,
    action == ActionType::Flip ? state.rows : state.columns,
    action == ActionType::Flip ? state.columns : state.rows,
  };
  validate_shape(transformed);

  using Piece = std::pair<int, std::uint8_t>;
  std::array<std::vector<Piece>, 10> target_columns;

  for (int column = 0; column < state.columns; column += 1) {
    for (int row = 0; row < state.rows; row += 1) {
      const Mask source = bit(state, column, row);
      std::uint8_t owner = 0;
      if ((state.mover & source) != 0) owner = 1;
      else if ((state.opponent & source) != 0) owner = 2;
      if (owner == 0) continue;

      int target_column = 0;
      int target_row = 0;
      if (action == ActionType::Flip) {
        target_column = column;
        target_row = state.rows - 1 - row;
      } else if (action == ActionType::RotateClockwise) {
        target_column = row;
        target_row = state.columns - 1 - column;
      } else {
        target_column = state.rows - 1 - row;
        target_row = column;
      }
      target_columns[target_column].push_back({target_row, owner});
    }
  }

  for (int column = 0; column < transformed.columns; column += 1) {
    auto& pieces = target_columns[column];
    std::sort(pieces.begin(), pieces.end());
    for (int row = 0; row < static_cast<int>(pieces.size()); row += 1) {
      const Mask target = bit(transformed, column, row);
      if (pieces[row].second == 1) transformed.mover |= target;
      else transformed.opponent |= target;
    }
  }

  return transformed;
}

Transition apply(const State& state, const Action& action, int connect) {
  State next = state;

  if (action.type == ActionType::Drop) {
    if (action.column >= state.columns) throw std::runtime_error("Drop column is out of range.");
    const Mask occupied = state.mover | state.opponent;
    int row = 0;
    while (row < state.rows && (occupied & bit(state, action.column, row)) != 0) row += 1;
    if (row >= state.rows) throw std::runtime_error("Drop action targets a full column.");
    next.mover |= bit(state, action.column, row);
    if (has_win(next.mover, stride(next), connect)) return {kWin, {}};
  } else {
    next = transform(state, action.type);
    const bool mover_won = has_win(next.mover, stride(next), connect);
    const bool opponent_won = has_win(next.opponent, stride(next), connect);
    if (mover_won && opponent_won) return {kLoss, {}};
    if (mover_won) return {kWin, {}};
    if (opponent_won) return {kLoss, {}};
  }

  if (is_full(next)) return {kDraw, {}};
  return {kPlaying, {next.opponent, next.mover, next.rows, next.columns}};
}

std::vector<Action> legal_actions(const State& state) {
  if (is_full(state)) return {};
  std::vector<Action> actions;
  const Mask occupied = state.mover | state.opponent;
  for (int column = 0; column < state.columns; column += 1) {
    if ((occupied & bit(state, column, state.rows - 1)) == 0) {
      actions.push_back({ActionType::Drop, static_cast<std::uint8_t>(column)});
    }
  }
  actions.push_back({ActionType::Flip, 0});
  actions.push_back({ActionType::RotateClockwise, 0});
  actions.push_back({ActionType::RotateCounterClockwise, 0});
  return actions;
}

bool same_action(const Action& first, const Action& second) {
  return first.type == second.type
    && (first.type != ActionType::Drop || first.column == second.column);
}

Graph build_graph(const State& root, int connect, std::uint32_t maximum_states) {
  if (connect < 1 || connect > std::max(root.rows, root.columns)) {
    throw std::runtime_error("Connect length does not fit the board.");
  }
  if (has_win(root.mover, stride(root), connect)
      || has_win(root.opponent, stride(root), connect)) {
    throw std::runtime_error("The root position is already won.");
  }
  const CanonicalState root_canonical = canonicalize(root);
  Graph graph;
  graph.root_mirrored = root_canonical.mirrored;
  graph.root_columns = root_canonical.state.columns;
  graph.nodes.push_back({root_canonical.state, {}, {}});

  std::unordered_map<Key, std::uint32_t, KeyHash> indices;
  indices.emplace(root_canonical.key, 0);

  for (std::uint32_t cursor = 0; cursor < graph.nodes.size(); cursor += 1) {
    const State state = graph.nodes[cursor].state;
    std::vector<std::int8_t> seen_terminal;
    std::vector<Key> seen_children;

    for (const Action& action : legal_actions(state)) {
      const Transition transition = apply(state, action, connect);
      if (transition.terminal != kPlaying) {
        if (std::find(seen_terminal.begin(), seen_terminal.end(), transition.terminal)
            != seen_terminal.end()) continue;
        seen_terminal.push_back(transition.terminal);
        graph.nodes[cursor].edges.push_back({
          action,
          transition.terminal,
          std::numeric_limits<std::uint32_t>::max(),
        });
        continue;
      }

      const CanonicalState child = canonicalize(transition.state);
      if (std::find(seen_children.begin(), seen_children.end(), child.key) != seen_children.end()) {
        continue;
      }
      seen_children.push_back(child.key);

      auto iterator = indices.find(child.key);
      if (iterator == indices.end()) {
        if (graph.nodes.size() >= maximum_states) {
          throw std::runtime_error("Native exact Chaos graph exceeded its state limit.");
        }
        const std::uint32_t child_index = static_cast<std::uint32_t>(graph.nodes.size());
        iterator = indices.emplace(child.key, child_index).first;
        graph.nodes.push_back({child.state, {}, {}});
      }
      graph.nodes[cursor].edges.push_back({action, kPlaying, iterator->second});
    }
  }

  for (std::uint32_t parent = 0; parent < graph.nodes.size(); parent += 1) {
    for (std::uint8_t edge = 0; edge < graph.nodes[parent].edges.size(); edge += 1) {
      const std::uint32_t child = graph.nodes[parent].edges[edge].next;
      if (child != std::numeric_limits<std::uint32_t>::max()) {
        graph.nodes[child].predecessors.push_back({parent, edge});
      }
    }
  }
  return graph;
}

Result solve(const State& root, int connect, std::uint32_t maximum_states = 2'000'000) {
  Graph graph = build_graph(root, connect, maximum_states);
  const std::size_t count = graph.nodes.size();
  std::vector<std::int8_t> values(count, kUnknown);
  std::vector<std::uint32_t> ranks(count, 0);
  std::vector<std::int32_t> best_edges(count, -1);
  std::vector<std::uint16_t> losing_actions(count, 0);
  std::vector<std::uint32_t> maximum_child_rank(count, 0);

  using QueueEntry = std::pair<std::uint32_t, std::uint32_t>;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;

  for (std::uint32_t index = 0; index < count; index += 1) {
    const Node& node = graph.nodes[index];
    int winning_edge = -1;
    int first_loss = -1;
    int losses = 0;
    for (int edge = 0; edge < static_cast<int>(node.edges.size()); edge += 1) {
      if (node.edges[edge].terminal == kWin && winning_edge < 0) winning_edge = edge;
      else if (node.edges[edge].terminal == kLoss) {
        losses += 1;
        if (first_loss < 0) first_loss = edge;
      }
    }
    losing_actions[index] = losses;
    if (winning_edge >= 0) {
      values[index] = kWin;
      ranks[index] = 1;
      best_edges[index] = winning_edge;
      queue.push({1, index});
    } else if (!node.edges.empty() && losses == static_cast<int>(node.edges.size())) {
      values[index] = kLoss;
      ranks[index] = 1;
      best_edges[index] = first_loss;
      queue.push({1, index});
    }
  }

  while (!queue.empty()) {
    const auto [rank, child] = queue.top();
    queue.pop();
    if (rank != ranks[child]) continue;

    for (const Predecessor& predecessor : graph.nodes[child].predecessors) {
      const std::uint32_t parent = predecessor.node;
      if (values[parent] != kUnknown) continue;
      if (values[child] == kLoss) {
        values[parent] = kWin;
        ranks[parent] = ranks[child] + 1;
        best_edges[parent] = predecessor.edge;
        queue.push({ranks[parent], parent});
      } else if (values[child] == kWin) {
        losing_actions[parent] += 1;
        if (ranks[child] >= maximum_child_rank[parent]) {
          maximum_child_rank[parent] = ranks[child];
          best_edges[parent] = predecessor.edge;
        }
        if (losing_actions[parent] == graph.nodes[parent].edges.size()) {
          values[parent] = kLoss;
          ranks[parent] = maximum_child_rank[parent] + 1;
          queue.push({ranks[parent], parent});
        }
      }
    }
  }

  Result result;
  result.states = static_cast<std::uint32_t>(count);
  for (std::uint32_t index = 0; index < count; index += 1) {
    if (values[index] == kUnknown) {
      values[index] = kDraw;
      result.draws += 1;
    } else if (values[index] == kWin) result.wins += 1;
    else result.losses += 1;
  }

  for (std::uint32_t index = 0; index < count; index += 1) {
    if (values[index] != kDraw) continue;
    for (int edge = 0; edge < static_cast<int>(graph.nodes[index].edges.size()); edge += 1) {
      const Edge& candidate = graph.nodes[index].edges[edge];
      if (candidate.terminal == kDraw
          || (candidate.terminal == kPlaying && values[candidate.next] == kDraw)) {
        best_edges[index] = edge;
        break;
      }
    }
  }

  result.value = values[0];
  result.rank = ranks[0];
  if (best_edges[0] >= 0) {
    result.action = graph.nodes[0].edges[best_edges[0]].action;
    if (graph.root_mirrored) result.action = reflect_action(result.action, graph.root_columns);
    result.has_action = true;
  }
  return result;
}

State from_board(const std::vector<std::vector<int>>& board, int current_player) {
  if (board.empty() || board[0].empty()) throw std::runtime_error("Board cannot be empty.");
  if (current_player != 1 && current_player != 2) {
    throw std::runtime_error("Current player must be 1 or 2.");
  }
  State state{0, 0, static_cast<std::uint8_t>(board.size()), static_cast<std::uint8_t>(board[0].size())};
  validate_shape(state);
  for (const auto& row : board) {
    if (row.size() != state.columns) throw std::runtime_error("Board must be rectangular.");
  }
  for (int column = 0; column < state.columns; column += 1) {
    bool found_piece = false;
    for (int top_row = 0; top_row < state.rows; top_row += 1) {
      int owner = board[top_row][column];
      if (owner == 0) {
        if (found_piece) throw std::runtime_error("Board pieces must obey gravity.");
        continue;
      }
      found_piece = true;
      if (owner != 1 && owner != 2) throw std::runtime_error("Invalid board cell.");
      if (current_player == 2) owner = owner == 1 ? 2 : 1;
      const int bottom_row = state.rows - 1 - top_row;
      if (owner == 1) state.mover |= bit(state, column, bottom_row);
      else state.opponent |= bit(state, column, bottom_row);
    }
  }
  return state;
}

std::string action_json(const Result& result) {
  if (!result.has_action) return "null";
  if (result.action.type == ActionType::Drop) {
    return "{\"type\":\"drop\",\"column\":" + std::to_string(result.action.column) + "}";
  }
  if (result.action.type == ActionType::Flip) return "{\"type\":\"flip\"}";
  if (result.action.type == ActionType::RotateClockwise) return "{\"type\":\"rotateCW\"}";
  return "{\"type\":\"rotateCCW\"}";
}

void require_case(
  const std::string& name,
  const State& state,
  int connect,
  std::int8_t expected_value,
  std::uint32_t expected_states,
  const Action* expected_action = nullptr
) {
  const Result result = solve(state, connect);
  if (result.value != expected_value) throw std::runtime_error(name + ": value mismatch.");
  if (result.states != expected_states) throw std::runtime_error(name + ": state-count mismatch.");
  if (expected_action != nullptr
      && (!result.has_action || !same_action(result.action, *expected_action))) {
    throw std::runtime_error(name + ": action mismatch.");
  }
  std::cout
    << "{\"name\":\"" << name << "\",\"value\":" << static_cast<int>(result.value)
    << ",\"states\":" << result.states << ",\"rank\":" << result.rank
    << ",\"action\":" << action_json(result) << "}" << '\n';
}

void verify() {
  require_case("2x2-connect2", from_board({{0, 0}, {0, 0}}, 1), 2, kWin, 6);
  require_case(
    "3x3-connect3",
    from_board({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 1),
    3,
    kDraw,
    628
  );
  const Action expected{ActionType::RotateClockwise, 0};
  require_case(
    "6x7-endgame-fixture",
    from_board({
      {1, 1, 1, 2, 1, 0, 0},
      {2, 2, 2, 1, 2, 0, 0},
      {2, 1, 2, 1, 2, 1, 0},
      {2, 1, 1, 1, 2, 2, 0},
      {1, 2, 2, 2, 1, 2, 2},
      {1, 1, 2, 2, 1, 1, 1},
    }, 1),
    4,
    kWin,
    2'585,
    &expected
  );
}

}  // namespace perfect_chaos

int main(int argc, char** argv) {
  try {
    if (argc != 2 || std::string(argv[1]) != "verify") {
      std::cerr << "Usage: perfect-chaos-native verify\n";
      return 2;
    }
    perfect_chaos::verify();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
