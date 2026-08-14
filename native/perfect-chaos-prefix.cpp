#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prefix {
using Mask = std::uint64_t;
constexpr std::uint32_t NO_NODE = std::numeric_limits<std::uint32_t>::max();

enum class ActionType : std::uint8_t { Drop = 0, Flip = 1, CW = 2, CCW = 3 };
enum class Terminal : std::uint8_t { Playing = 0, AiWin = 1, Draw = 2, AiLoss = 3 };
struct Action { ActionType type = ActionType::Flip; std::uint8_t column = 0; };
struct State {
  Mask mover = 0;
  Mask opponent = 0;
  std::uint8_t rows = 6;
  std::uint8_t columns = 7;
  bool ai_turn = true;
};
struct Key {
  Mask mover = 0;
  Mask opponent = 0;
  std::uint8_t rows = 0;
  std::uint8_t columns = 0;
  bool ai_turn = false;
  bool operator==(const Key& other) const {
    return mover == other.mover && opponent == other.opponent
      && rows == other.rows && columns == other.columns && ai_turn == other.ai_turn;
  }
};
struct KeyHash {
  std::size_t operator()(const Key& key) const {
    std::uint64_t x = key.mover * 0x9e3779b97f4a7c15ULL;
    x ^= key.opponent + 0x517cc1b727220a95ULL;
    x ^= (static_cast<std::uint64_t>(key.rows) << 16U)
       | (static_cast<std::uint64_t>(key.columns) << 8U)
       | static_cast<std::uint64_t>(key.ai_turn);
    x ^= x >> 30U; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31U;
    return static_cast<std::size_t>(x);
  }
};
struct Canonical { State state; Key key; bool mirrored = false; };
struct Transition { int mover_result = 0; State state; }; // 0 playing, 1 win, -1 loss, 2 draw
struct Edge { std::uint32_t next = NO_NODE; Terminal terminal = Terminal::Playing; Action action; };
struct Node { State state; std::uint32_t first_edge = 0; std::uint16_t edge_count = 0; bool frontier = false; };
struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<std::uint32_t> roots;
  std::vector<std::uint32_t> pred_offsets;
  std::vector<std::uint32_t> predecessors;
};
struct Summary {
  std::uint32_t input_roots = 0;
  std::uint32_t graph_states = 0;
  std::uint64_t graph_edges = 0;
  std::uint32_t losing_states = 0;
  std::uint32_t safe_states = 0;
  std::uint32_t closure_states = 0;
  std::uint32_t closure_ai_states = 0;
  std::uint32_t closure_opponent_states = 0;
  std::uint32_t frontier_states = 0;
  std::uint64_t terminal_ai_wins = 0;
  std::uint64_t terminal_draws = 0;
  std::uint64_t revisited_edges = 0;
};
struct Closure {
  std::vector<std::uint32_t> states;
  std::vector<std::uint32_t> frontier;
  std::vector<std::pair<std::uint32_t, Action>> policy;
  Summary summary;
};

int stride(const State& s) { return static_cast<int>(s.rows) + 1; }
Mask bit(const State& s, int c, int r) { return Mask{1} << (c * stride(s) + r); }
void validate(const State& s) {
  if (!s.rows || !s.columns) throw std::runtime_error("Board dimensions must be positive.");
  if (static_cast<int>(s.columns) * (static_cast<int>(s.rows) + 1) > 63)
    throw std::runtime_error("Board does not fit the native sentinel mask.");
  if ((s.mover & s.opponent) != 0) throw std::runtime_error("Mover and opponent masks overlap.");
}
std::uint8_t pieces(const State& s) { return static_cast<std::uint8_t>(std::popcount(s.mover | s.opponent)); }
bool has_win(Mask p, int st, int connect) {
  for (int shift : std::array<int,4>{1, st, st - 1, st + 1}) {
    Mask run = p;
    for (int i = 1; i < connect; ++i) run &= p >> (i * shift);
    if (run) return true;
  }
  return false;
}
bool full(const State& s) {
  Mask occupied = s.mover | s.opponent;
  for (int c = 0; c < s.columns; ++c)
    if ((occupied & bit(s, c, s.rows - 1)) == 0) return false;
  return true;
}
void validate_table_state(const State& s) {
  validate(s);
  if (!((s.rows == 6 && s.columns == 7) || (s.rows == 7 && s.columns == 6)))
    throw std::runtime_error("Perfect Chaos tables require a 6x7 or 7x6 board.");
  const Mask occupied = s.mover | s.opponent;
  const int st = stride(s);
  const int used_bits = static_cast<int>(s.columns) * st;
  if ((occupied >> used_bits) != 0) throw std::runtime_error("Perfect Chaos table sets a bit outside the board.");
  const Mask column_mask = (Mask{1} << s.rows) - 1;
  for (int column = 0; column < s.columns; ++column) {
    if ((occupied & bit(s,column,s.rows)) != 0) throw std::runtime_error("Perfect Chaos table sets a sentinel bit.");
    const Mask column_bits = (occupied >> (column * st)) & column_mask;
    if ((column_bits & (column_bits + 1)) != 0) throw std::runtime_error("Perfect Chaos table violates gravity.");
  }
  if (has_win(s.mover,st,4) || has_win(s.opponent,st,4) || full(s))
    throw std::runtime_error("Perfect Chaos table contains a terminal state.");
}
Mask mirror_mask(Mask mask, int rows, int columns) {
  const int st = rows + 1;
  const Mask column_mask = (Mask{1} << rows) - 1;
  Mask reflected = 0;
  for (int c = 0; c < columns; ++c) {
    reflected |= ((mask >> (c * st)) & column_mask)
      << ((columns - 1 - c) * st);
  }
  return reflected;
}
State mirror(const State& s) {
  return {
    mirror_mask(s.mover, s.rows, s.columns),
    mirror_mask(s.opponent, s.rows, s.columns),
    s.rows,
    s.columns,
    s.ai_turn,
  };
}
Canonical canonicalize(const State& s) {
  validate(s);
  State reflected = mirror(s);
  bool use = reflected.mover < s.mover
    || (reflected.mover == s.mover && reflected.opponent < s.opponent);
  State chosen = use ? reflected : s;
  return {chosen, {chosen.mover, chosen.opponent, chosen.rows, chosen.columns, chosen.ai_turn}, use};
}
State transform(const State& s, ActionType type) {
  if (type == ActionType::Drop) throw std::runtime_error("Drop is not a transformation.");
  State out{0,0, type == ActionType::Flip ? s.rows : s.columns,
                  type == ActionType::Flip ? s.columns : s.rows, s.ai_turn};
  validate(out);
  using Piece = std::pair<int,std::uint8_t>;
  std::array<std::vector<Piece>,10> columns;
  for (int c = 0; c < s.columns; ++c) for (int r = 0; r < s.rows; ++r) {
    Mask source = bit(s,c,r); std::uint8_t owner = 0;
    if (s.mover & source) owner = 1; else if (s.opponent & source) owner = 2;
    if (!owner) continue;
    int tc = 0, tr = 0;
    if (type == ActionType::Flip) { tc = c; tr = s.rows - 1 - r; }
    else if (type == ActionType::CW) { tc = r; tr = s.columns - 1 - c; }
    else { tc = s.rows - 1 - r; tr = c; }
    columns[tc].push_back({tr,owner});
  }
  for (int c = 0; c < out.columns; ++c) {
    auto& col = columns[c]; std::sort(col.begin(),col.end());
    for (int r = 0; r < static_cast<int>(col.size()); ++r) {
      Mask target = bit(out,c,r);
      if (col[r].second == 1) out.mover |= target; else out.opponent |= target;
    }
  }
  return out;
}
Transition apply(const State& s, Action a, int connect) {
  State next = s;
  if (a.type == ActionType::Drop) {
    if (a.column >= s.columns) throw std::runtime_error("Drop column out of range.");
    Mask occupied = s.mover | s.opponent; int r = 0;
    while (r < s.rows && (occupied & bit(s,a.column,r))) ++r;
    if (r >= s.rows) throw std::runtime_error("Drop targets a full column.");
    next.mover |= bit(s,a.column,r);
    if (has_win(next.mover,stride(next),connect)) return {1,{}};
  } else {
    next = transform(s,a.type);
    bool mover_won = has_win(next.mover,stride(next),connect);
    bool opponent_won = has_win(next.opponent,stride(next),connect);
    if (mover_won && opponent_won) return {-1,{}};
    if (mover_won) return {1,{}};
    if (opponent_won) return {-1,{}};
  }
  if (full(next)) return {2,{}};
  return {0,{next.opponent,next.mover,next.rows,next.columns,!s.ai_turn}};
}
std::vector<Action> legal(const State& s) {
  if (full(s)) return {};
  std::vector<int> order(s.columns);
  for (int c = 0; c < s.columns; ++c) order[c] = c;
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){
    int da = std::abs(2*a-(s.columns-1)), db = std::abs(2*b-(s.columns-1));
    return da != db ? da < db : a < b;
  });
  std::vector<Action> actions; Mask occupied = s.mover | s.opponent;
  for (int c : order) if ((occupied & bit(s,c,s.rows-1)) == 0)
    actions.push_back({ActionType::Drop,static_cast<std::uint8_t>(c)});
  actions.push_back({ActionType::Flip,0});
  actions.push_back({ActionType::CW,0});
  actions.push_back({ActionType::CCW,0});
  return actions;
}
Terminal terminal_for_ai(const State& s, int mover_result) {
  if (mover_result == 2) return Terminal::Draw;
  bool mover_won = mover_result == 1;
  return (mover_won == s.ai_turn) ? Terminal::AiWin : Terminal::AiLoss;
}

Graph build_graph(const std::vector<State>& input_roots, int connect,
                  std::uint8_t frontier_pieces, std::uint32_t maximum_states) {
  if (input_roots.empty()) throw std::runtime_error("At least one root is required.");
  Graph g;
  g.nodes.reserve(std::min<std::uint32_t>(maximum_states,1'500'000));
  g.edges.reserve(std::min<std::uint64_t>(static_cast<std::uint64_t>(maximum_states)*4ULL,8'000'000ULL));
  std::unordered_map<Key,std::uint32_t,KeyHash> ids;
  ids.reserve(std::min<std::uint32_t>(maximum_states,2'000'000));
  for (const State& raw : input_roots) {
    Canonical root = canonicalize(raw);
    if (pieces(root.state) > frontier_pieces) throw std::runtime_error("Root lies beyond the target frontier.");
    auto [it,inserted] = ids.emplace(root.key,static_cast<std::uint32_t>(g.nodes.size()));
    if (inserted) g.nodes.push_back({root.state,0,0,pieces(root.state) >= frontier_pieces});
    g.roots.push_back(it->second);
  }
  std::sort(g.roots.begin(),g.roots.end());
  g.roots.erase(std::unique(g.roots.begin(),g.roots.end()),g.roots.end());

  for (std::uint32_t cursor = 0; cursor < g.nodes.size(); ++cursor) {
    State state = g.nodes[cursor].state;
    bool boundary = g.nodes[cursor].frontier;
    std::uint32_t first = static_cast<std::uint32_t>(g.edges.size());
    g.nodes[cursor].first_edge = first;
    if (boundary) continue;
    std::array<bool,4> seen_terminal{};
    std::vector<Key> seen_children; seen_children.reserve(10);
    for (Action action : legal(state)) {
      Transition tr = apply(state,action,connect);
      if (tr.mover_result != 0) {
        Terminal t = terminal_for_ai(state,tr.mover_result);
        std::size_t ti = static_cast<std::size_t>(t);
        if (seen_terminal[ti]) continue;
        seen_terminal[ti] = true;
        g.edges.push_back({NO_NODE,t,action});
        continue;
      }
      Canonical child = canonicalize(tr.state);
      if (std::find(seen_children.begin(),seen_children.end(),child.key) != seen_children.end()) continue;
      seen_children.push_back(child.key);
      auto it = ids.find(child.key);
      if (it == ids.end()) {
        if (g.nodes.size() >= maximum_states) throw std::runtime_error("Prefix graph exceeded its state limit.");
        std::uint32_t id = static_cast<std::uint32_t>(g.nodes.size());
        it = ids.emplace(child.key,id).first;
        g.nodes.push_back({child.state,0,0,pieces(child.state) >= frontier_pieces});
      }
      g.edges.push_back({it->second,Terminal::Playing,action});
    }
    std::size_t count = g.edges.size() - first;
    if (count > std::numeric_limits<std::uint16_t>::max()) throw std::runtime_error("Too many actions.");
    g.nodes[cursor].edge_count = static_cast<std::uint16_t>(count);
  }

  std::vector<std::uint32_t> incoming(g.nodes.size(),0);
  for (const Edge& e : g.edges) if (e.terminal == Terminal::Playing) ++incoming[e.next];
  g.pred_offsets.resize(g.nodes.size()+1,0);
  for (std::size_t i=0;i<incoming.size();++i) g.pred_offsets[i+1]=g.pred_offsets[i]+incoming[i];
  g.predecessors.resize(g.pred_offsets.back());
  std::vector<std::uint32_t> write = g.pred_offsets;
  for (std::uint32_t p=0;p<g.nodes.size();++p) {
    const Node& n = g.nodes[p];
    for (std::uint32_t o=0;o<n.edge_count;++o) {
      const Edge& e = g.edges[n.first_edge+o];
      if (e.terminal == Terminal::Playing) g.predecessors[write[e.next]++] = p;
    }
  }
  return g;
}

using KeySet = std::unordered_set<Key, KeyHash>;
Key key_of(const State& s) { return {s.mover, s.opponent, s.rows, s.columns, s.ai_turn}; }

std::vector<std::uint8_t> solve_losing(const Graph& g, const KeySet& rejected_frontier = {}) {
  std::vector<std::uint8_t> losing(g.nodes.size(),0);
  std::vector<std::uint16_t> bad(g.nodes.size(),0);
  std::queue<std::uint32_t> q;
  auto mark=[&](std::uint32_t i){ if(!losing[i]){losing[i]=1;q.push(i);} };
  for (std::uint32_t i=0;i<g.nodes.size();++i) {
    const Node& n=g.nodes[i];
    if (n.frontier) {
      if (rejected_frontier.contains(key_of(n.state))) mark(i);
      continue;
    }
    for (std::uint32_t o=0;o<n.edge_count;++o) if(g.edges[n.first_edge+o].terminal==Terminal::AiLoss){
      ++bad[i]; if(!n.state.ai_turn){mark(i);break;}
    }
    if(n.state.ai_turn && n.edge_count && bad[i]==n.edge_count) mark(i);
  }
  while(!q.empty()){
    std::uint32_t child=q.front();q.pop();
    for(std::uint32_t k=g.pred_offsets[child];k<g.pred_offsets[child+1];++k){
      std::uint32_t parent=g.predecessors[k]; if(losing[parent]) continue;
      const Node& n=g.nodes[parent];
      if(!n.state.ai_turn) mark(parent);
      else if(++bad[parent]==n.edge_count) mark(parent);
    }
  }
  return losing;
}

int priority(const Graph& g,const Edge& e,const Node& parent,const std::vector<std::uint8_t>& losing){
  if(e.terminal==Terminal::AiWin) return 0;
  if(e.terminal==Terminal::AiLoss) return 100;
  if(e.terminal==Terminal::Draw) return 2;
  if(losing[e.next]) return 100;
  const Node& child=g.nodes[e.next];
  if(pieces(child.state)>pieces(parent.state)) return child.frontier?1:2;
  return 3;
}
std::vector<std::int16_t> select_policy(const Graph& g,const std::vector<std::uint8_t>& losing){
  std::vector<std::int16_t> policy(g.nodes.size(),-1);
  for(std::uint32_t i=0;i<g.nodes.size();++i){
    const Node& n=g.nodes[i]; if(!n.state.ai_turn||n.frontier||losing[i]) continue;
    int best=1000,best_o=-1;
    for(std::uint32_t o=0;o<n.edge_count;++o){int p=priority(g,g.edges[n.first_edge+o],n,losing);if(p<best){best=p;best_o=o;}}
    if(best_o<0||best>=100) throw std::runtime_error("Safe AI state has no safe action.");
    policy[i]=static_cast<std::int16_t>(best_o);
  }
  return policy;
}
Closure close_policy(const Graph& g,const std::vector<std::uint8_t>& losing,const std::vector<std::int16_t>& policy){
  Closure c; std::vector<std::uint8_t> seen(g.nodes.size(),0); std::queue<std::uint32_t> q;
  for(std::uint32_t root:g.roots){if(losing[root])throw std::runtime_error("An input root is losing before the target frontier.");if(!seen[root]){seen[root]=1;q.push(root);}}
  auto follow=[&](const Edge& e){
    if(e.terminal==Terminal::AiLoss) throw std::runtime_error("Policy closure reaches an AI loss.");
    if(e.terminal==Terminal::AiWin){++c.summary.terminal_ai_wins;return;}
    if(e.terminal==Terminal::Draw){++c.summary.terminal_draws;return;}
    if(losing[e.next]) throw std::runtime_error("Policy selected a losing child.");
    if(seen[e.next]) ++c.summary.revisited_edges; else {seen[e.next]=1;q.push(e.next);}
  };
  while(!q.empty()){
    std::uint32_t i=q.front();q.pop();c.states.push_back(i);
    const Node& n=g.nodes[i]; if(n.frontier){c.frontier.push_back(i);continue;}
    if(n.state.ai_turn){
      ++c.summary.closure_ai_states; int o=policy[i];
      if(o<0||o>=n.edge_count) throw std::runtime_error("Missing policy action.");
      const Edge& e=g.edges[n.first_edge+static_cast<std::uint32_t>(o)];
      c.policy.push_back({i,e.action});follow(e);
    } else {
      ++c.summary.closure_opponent_states;
      for(std::uint32_t o=0;o<n.edge_count;++o) follow(g.edges[n.first_edge+o]);
    }
  }
  c.summary.input_roots=g.roots.size();c.summary.graph_states=g.nodes.size();c.summary.graph_edges=g.edges.size();
  c.summary.closure_states=c.states.size();c.summary.frontier_states=c.frontier.size();
  for(std::uint8_t v:losing) v?++c.summary.losing_states:++c.summary.safe_states;
  return c;
}

void write_u64(std::ostream& o,std::uint64_t v){for(int s=0;s<64;s+=8)o.put(static_cast<char>((v>>s)&255));}
void write_u32(std::ostream& o,std::uint32_t v){for(int s=0;s<32;s+=8)o.put(static_cast<char>((v>>s)&255));}
std::uint64_t read_u64(std::istream& in){std::uint64_t v=0;for(int s=0;s<64;s+=8){int c=in.get();if(c<0)throw std::runtime_error("Truncated binary file.");v|=static_cast<std::uint64_t>(static_cast<unsigned char>(c))<<s;}return v;}
std::uint32_t read_u32(std::istream& in){std::uint32_t v=0;for(int s=0;s<32;s+=8){int c=in.get();if(c<0)throw std::runtime_error("Truncated binary file.");v|=static_cast<std::uint32_t>(static_cast<unsigned char>(c))<<s;}return v;}
void header(std::ostream& o,const std::array<char,8>& magic,std::uint8_t role,std::uint8_t boundary,std::uint32_t count,std::uint8_t size){o.write(magic.data(),8);o.put(1);o.put(role);o.put(boundary);o.put(size);write_u32(o,count);}
bool state_less(const State& a, const State& b) {
  if (a.rows != b.rows) return a.rows < b.rows;
  if (a.columns != b.columns) return a.columns < b.columns;
  if (a.ai_turn != b.ai_turn) return a.ai_turn < b.ai_turn;
  if (a.mover != b.mover) return a.mover < b.mover;
  return a.opponent < b.opponent;
}
void write_policy(const std::string& path,const Graph&g,const Closure&c,std::uint8_t role,std::uint8_t boundary){
  auto rec=c.policy;std::sort(rec.begin(),rec.end(),[&](auto&a,auto&b){return state_less(g.nodes[a.first].state,g.nodes[b.first].state);});
  std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("Cannot create policy file.");header(o,{'C','4','C','P','O','L','1','\0'},role,boundary,rec.size(),20);
  for(auto [i,a]:rec){const State&s=g.nodes[i].state;write_u64(o,s.mover);write_u64(o,s.opponent);o.put(s.rows);o.put(s.columns);o.put(static_cast<char>(a.type));o.put(a.column);}if(!o)throw std::runtime_error("Policy write failed.");
}
void write_frontier(const std::string& path,const Graph&g,const Closure&c,std::uint8_t role,std::uint8_t boundary){
  auto rec=c.frontier;std::sort(rec.begin(),rec.end(),[&](auto a,auto b){return state_less(g.nodes[a].state,g.nodes[b].state);});
  std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("Cannot create frontier file.");header(o,{'C','4','C','F','R','N','1','\0'},role,boundary,rec.size(),19);
  for(auto i:rec){const State&s=g.nodes[i].state;write_u64(o,s.mover);write_u64(o,s.opponent);o.put(s.rows);o.put(s.columns);o.put(s.ai_turn?1:0);}if(!o)throw std::runtime_error("Frontier write failed.");
}
void write_policy_records(const std::string& path,std::vector<std::pair<State,Action>> records,std::uint8_t role,std::uint8_t boundary){
  std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){return state_less(a.first,b.first);});
  for(std::size_t i=1;i<records.size();++i){
    if(!state_less(records[i-1].first,records[i].first))throw std::runtime_error("Sliced policy records are not unique.");
  }
  std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("Cannot create sliced policy file.");
  header(o,{'C','4','C','P','O','L','1','\0'},role,boundary,static_cast<std::uint32_t>(records.size()),20);
  for(const auto& [state,action]:records){write_u64(o,state.mover);write_u64(o,state.opponent);o.put(state.rows);o.put(state.columns);o.put(static_cast<char>(action.type));o.put(action.column);}
  if(!o)throw std::runtime_error("Sliced policy write failed.");
}
void write_frontier_states(const std::string& path,std::vector<State> states,std::uint8_t role,std::uint8_t boundary){
  std::sort(states.begin(),states.end(),state_less);
  states.erase(std::unique(states.begin(),states.end(),[](const State& a,const State& b){return key_of(a)==key_of(b);}),states.end());
  std::ofstream o(path,std::ios::binary);if(!o)throw std::runtime_error("Cannot create sliced frontier file.");
  header(o,{'C','4','C','F','R','N','1','\0'},role,boundary,static_cast<std::uint32_t>(states.size()),19);
  for(const State& state:states){write_u64(o,state.mover);write_u64(o,state.opponent);o.put(state.rows);o.put(state.columns);o.put(state.ai_turn?1:0);}
  if(!o)throw std::runtime_error("Sliced frontier write failed.");
}
struct FrontierInput { std::uint8_t role=0; std::uint8_t boundary=0; std::vector<State> states; };
FrontierInput read_frontier(const std::string& path){
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open input frontier.");
  std::array<char,8> magic{};in.read(magic.data(),8);if(magic!=std::array<char,8>{'C','4','C','F','R','N','1','\0'})throw std::runtime_error("Invalid frontier magic.");
  int version=in.get(),role=in.get(),boundary=in.get(),record=in.get();if(version!=1||record!=19||role<1||role>2||boundary>42)throw std::runtime_error("Unsupported frontier header.");
  std::uint32_t count=read_u32(in);FrontierInput out{static_cast<std::uint8_t>(role),static_cast<std::uint8_t>(boundary),{}};out.states.reserve(count);
  State previous{};bool have=false;
  for(std::uint32_t i=0;i<count;++i){State s; s.mover=read_u64(in);s.opponent=read_u64(in);int r=in.get(),c=in.get(),a=in.get();if(r<0||c<0||a<0)throw std::runtime_error("Truncated frontier record.");s.rows=r;s.columns=c;s.ai_turn=a!=0;validate_table_state(s);if(pieces(s)!=out.boundary)throw std::runtime_error("Frontier record has the wrong piece count.");Canonical canon=canonicalize(s);if(!(canon.key==Key{s.mover,s.opponent,s.rows,s.columns,s.ai_turn}))throw std::runtime_error("Frontier record is not canonical.");if(have&&!state_less(previous,s))throw std::runtime_error("Frontier records are not strictly sorted.");previous=s;have=true;out.states.push_back(s);}if(in.peek()!=EOF)throw std::runtime_error("Frontier file has trailing bytes.");return out;
}
struct PolicyInput { std::uint8_t role=0; std::uint8_t boundary=0; std::unordered_map<Key,Action,KeyHash> actions; };
PolicyInput read_policy(const std::string& path){
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open input policy.");
  std::array<char,8> magic{};in.read(magic.data(),8);if(magic!=std::array<char,8>{'C','4','C','P','O','L','1','\0'})throw std::runtime_error("Invalid policy magic.");
  int version=in.get(),role=in.get(),boundary=in.get(),record=in.get();if(version!=1||record!=20||role<1||role>2||boundary>42)throw std::runtime_error("Unsupported policy header.");
  std::uint32_t count=read_u32(in);PolicyInput out{static_cast<std::uint8_t>(role),static_cast<std::uint8_t>(boundary),{}};out.actions.reserve(count);
  State previous{};bool have=false;
  for(std::uint32_t i=0;i<count;++i){
    State state;state.mover=read_u64(in);state.opponent=read_u64(in);int rows=in.get(),columns=in.get(),type=in.get(),column=in.get();
    if(rows<0||columns<0||type<0||column<0)throw std::runtime_error("Truncated policy record.");
    state.rows=static_cast<std::uint8_t>(rows);state.columns=static_cast<std::uint8_t>(columns);state.ai_turn=true;validate_table_state(state);
    if(pieces(state)>=out.boundary)throw std::runtime_error("Policy record lies at or beyond its frontier.");
    Canonical canon=canonicalize(state);if(!(canon.key==key_of(state)))throw std::runtime_error("Policy record is not canonical.");
    if(have&&!state_less(previous,state))throw std::runtime_error("Policy records are not strictly sorted.");
    previous=state;have=true;
    if(type>static_cast<int>(ActionType::CCW))throw std::runtime_error("Policy action type is invalid.");
    Action action{static_cast<ActionType>(type),static_cast<std::uint8_t>(column)};
    if(action.type==ActionType::Drop){
      if(action.column>=state.columns||(state.mover|state.opponent)&bit(state,action.column,state.rows-1))throw std::runtime_error("Policy drop is illegal.");
    }else if(action.column!=0)throw std::runtime_error("Policy transformation column must be zero.");
    if(!out.actions.emplace(key_of(state),action).second)throw std::runtime_error("Policy contains duplicate states.");
  }
  if(in.peek()!=EOF)throw std::runtime_error("Policy file has trailing bytes.");
  return out;
}

std::uint32_t number(const std::string&s,const std::string&label){std::size_t n=0;unsigned long v=std::stoul(s,&n);if(n!=s.size()||v>std::numeric_limits<std::uint32_t>::max())throw std::runtime_error(label+" is invalid.");return v;}
std::string role_name(std::uint8_t role){return role==1?"red":"yellow";}
void print(const Summary&s,std::uint8_t role,int from,int to){
  std::cout<<"{\"format\":\"connect4-chaos-prefix-certificate-v1\",\"role\":\""<<role_name(role)<<"\",\"fromPieces\":"<<static_cast<int>(from)<<",\"frontierPieces\":"<<static_cast<int>(to)
    <<",\"inputRoots\":"<<s.input_roots<<",\"graphStates\":"<<s.graph_states<<",\"graphEdges\":"<<s.graph_edges<<",\"losingStates\":"<<s.losing_states<<",\"safeStates\":"<<s.safe_states
    <<",\"closureStates\":"<<s.closure_states<<",\"closureAiStates\":"<<s.closure_ai_states<<",\"closureOpponentStates\":"<<s.closure_opponent_states<<",\"frontierStates\":"<<s.frontier_states
    <<",\"terminalAiWins\":"<<s.terminal_ai_wins<<",\"terminalDraws\":"<<s.terminal_draws<<",\"revisitedEdges\":"<<s.revisited_edges<<"}\n";
}
bool run_segment(const std::vector<State>& roots,std::uint8_t role,std::uint8_t from,std::uint8_t to,std::uint32_t max,const std::string&policy_path,const std::string&frontier_path,const KeySet& rejected_boundary={},const std::string& rejected_roots_path={}){
  if(to<=from)throw std::runtime_error("Target frontier must be beyond the input boundary.");
  Graph g=build_graph(roots,4,to,max);
  auto losing=solve_losing(g,rejected_boundary);
  std::vector<std::uint32_t> bad_roots;
  for(std::uint32_t root:g.roots) if(losing[root]) bad_roots.push_back(root);
  if(!bad_roots.empty()){
    if(!rejected_roots_path.empty()){
      Closure rejected; rejected.frontier=bad_roots;
      write_frontier(rejected_roots_path,g,rejected,role,from);
      std::cout<<"{\"format\":\"connect4-chaos-prefix-rejection-v1\",\"role\":\""<<role_name(role)<<"\",\"fromPieces\":"<<static_cast<int>(from)<<",\"frontierPieces\":"<<static_cast<int>(to)<<",\"losingInputRoots\":"<<bad_roots.size()<<"}\n";
      return false;
    }
    throw std::runtime_error("An input root is losing before the target frontier.");
  }
  auto policy=select_policy(g,losing);Closure c=close_policy(g,losing,policy);
  write_policy(policy_path,g,c,role,to);write_frontier(frontier_path,g,c,role,to);print(c.summary,role,from,to);
  return true;
}
void command_generate(int argc,char**argv){
  std::uint8_t role=1,to=8;std::uint32_t max=5'000'000;std::string policy,frontier,reject_path;
  for(int i=2;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error(a+" needs a value.");return std::string(argv[i]);};
    if(a=="--role"){std::string v=val();role=v=="red"?1:v=="yellow"?2:0;if(!role)throw std::runtime_error("Role must be red or yellow.");}
    else if(a=="--frontier-pieces")to=number(val(),"frontier-pieces");else if(a=="--maximum-states")max=number(val(),"maximum-states");else if(a=="--policy")policy=val();else if(a=="--frontier")frontier=val();else if(a=="--reject-frontier")reject_path=val();else throw std::runtime_error("Unknown argument: "+a);
  }
  if(policy.empty()||frontier.empty())throw std::runtime_error("--policy and --frontier are required.");
  KeySet rejected;
  if(!reject_path.empty()){
    FrontierInput input=read_frontier(reject_path);
    if(input.role!=role||input.boundary!=to)throw std::runtime_error("Rejected frontier role or boundary does not match generation target.");
    for(const State& state:input.states) rejected.insert(key_of(state));
  }
  run_segment({State{0,0,6,7,role==1}},role,0,to,max,policy,frontier,rejected);
}
void command_extend(int argc,char**argv){
  std::uint8_t to=0;std::uint32_t max=10'000'000;std::string input,policy,frontier,rejected,reject_boundary_path;
  for(int i=2;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error(a+" needs a value.");return std::string(argv[i]);};
    if(a=="--input-frontier")input=val();else if(a=="--frontier-pieces")to=number(val(),"frontier-pieces");else if(a=="--maximum-states")max=number(val(),"maximum-states");else if(a=="--policy")policy=val();else if(a=="--frontier")frontier=val();else if(a=="--rejected")rejected=val();else if(a=="--reject-frontier")reject_boundary_path=val();else throw std::runtime_error("Unknown argument: "+a);
  }
  if(input.empty()||policy.empty()||frontier.empty()||!to)throw std::runtime_error("extend requires input, target, policy and frontier paths.");
  FrontierInput f=read_frontier(input);
  KeySet rejected_boundary;
  if(!reject_boundary_path.empty()){
    FrontierInput bad=read_frontier(reject_boundary_path);
    if(bad.role!=f.role||bad.boundary!=to)throw std::runtime_error("Rejected frontier role or boundary does not match extension target.");
    for(const State& state:bad.states) rejected_boundary.insert(key_of(state));
  }
  if(!run_segment(f.states,f.role,f.boundary,to,max,policy,frontier,rejected_boundary,rejected)) throw std::runtime_error("Input frontier contains losing roots; rejection file written.");
}
void command_slice(int argc,char**argv){
  std::string input_path,policy_path,reference_path,output_policy,output_frontier;
  for(int i=2;i<argc;++i){std::string arg=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error(arg+" needs a value.");return std::string(argv[i]);};
    if(arg=="--input-frontier")input_path=value();else if(arg=="--policy")policy_path=value();else if(arg=="--reference-frontier")reference_path=value();else if(arg=="--output-policy")output_policy=value();else if(arg=="--output-frontier")output_frontier=value();else throw std::runtime_error("Unknown argument: "+arg);
  }
  if(input_path.empty()||policy_path.empty()||reference_path.empty()||output_policy.empty()||output_frontier.empty())throw std::runtime_error("slice requires input, policy, reference frontier, and output paths.");
  FrontierInput input=read_frontier(input_path);PolicyInput policy=read_policy(policy_path);FrontierInput reference=read_frontier(reference_path);
  if(input.role!=policy.role||input.role!=reference.role||policy.boundary!=reference.boundary||input.boundary>=policy.boundary)throw std::runtime_error("Slice table metadata does not align.");
  KeySet reference_keys;reference_keys.reserve(reference.states.size());for(const State& state:reference.states)reference_keys.insert(key_of(state));
  KeySet visited;visited.reserve(input.states.size()+policy.actions.size()+reference.states.size());std::queue<State> queue;
  for(const State& raw:input.states){Canonical root=canonicalize(raw);if(visited.insert(root.key).second)queue.push(root.state);}
  std::vector<std::pair<State,Action>> used_policy;used_policy.reserve(policy.actions.size());std::vector<State> boundary;boundary.reserve(reference.states.size());
  std::uint64_t terminal_ai_wins=0,terminal_draws=0,revisited_edges=0;std::uint32_t ai_states=0,opponent_states=0;
  auto follow=[&](const State& parent,const Transition& transition){
    if(transition.mover_result!=0){Terminal terminal=terminal_for_ai(parent,transition.mover_result);if(terminal==Terminal::AiLoss)throw std::runtime_error("Sliced policy reaches an AI loss.");if(terminal==Terminal::AiWin)++terminal_ai_wins;else if(terminal==Terminal::Draw)++terminal_draws;return;}
    Canonical child=canonicalize(transition.state);if(!visited.insert(child.key).second)++revisited_edges;else queue.push(child.state);
  };
  while(!queue.empty()){
    State state=queue.front();queue.pop();std::uint8_t count=pieces(state);
    if(count==policy.boundary){if(!reference_keys.contains(key_of(state)))throw std::runtime_error("Sliced policy reaches a frontier state outside the reference certificate.");boundary.push_back(state);continue;}
    if(count>policy.boundary)throw std::runtime_error("Sliced policy crossed its target frontier.");
    if(state.ai_turn){
      ++ai_states;auto found=policy.actions.find(key_of(state));if(found==policy.actions.end())throw std::runtime_error("Sliced policy is missing a reachable AI state.");
      used_policy.push_back({state,found->second});follow(state,apply(state,found->second,4));
    }else{
      ++opponent_states;std::array<bool,4> seen_terminal{};std::vector<Key> seen_children;seen_children.reserve(10);
      for(Action action:legal(state)){
        Transition transition=apply(state,action,4);
        if(transition.mover_result!=0){Terminal terminal=terminal_for_ai(state,transition.mover_result);std::size_t index=static_cast<std::size_t>(terminal);if(seen_terminal[index])continue;seen_terminal[index]=true;follow(state,transition);continue;}
        Canonical child=canonicalize(transition.state);if(std::find(seen_children.begin(),seen_children.end(),child.key)!=seen_children.end())continue;seen_children.push_back(child.key);
        if(!visited.insert(child.key).second)++revisited_edges;else queue.push(child.state);
      }
    }
  }
  write_policy_records(output_policy,std::move(used_policy),input.role,policy.boundary);write_frontier_states(output_frontier,std::move(boundary),input.role,policy.boundary);
  FrontierInput sliced_frontier=read_frontier(output_frontier);PolicyInput sliced_policy=read_policy(output_policy);
  std::cout<<"{\"format\":\"connect4-chaos-policy-slice-v1\",\"role\":\""<<role_name(input.role)<<"\",\"fromPieces\":"<<static_cast<int>(input.boundary)<<",\"targetPieces\":"<<static_cast<int>(policy.boundary)
    <<",\"inputRoots\":"<<input.states.size()<<",\"sourcePolicyEntries\":"<<policy.actions.size()<<",\"sourceFrontierStates\":"<<reference.states.size()<<",\"policyEntries\":"<<sliced_policy.actions.size()<<",\"frontierStates\":"<<sliced_frontier.states.size()
    <<",\"closureStates\":"<<visited.size()<<",\"aiStates\":"<<ai_states<<",\"opponentStates\":"<<opponent_states<<",\"terminalAiWins\":"<<terminal_ai_wins<<",\"terminalDraws\":"<<terminal_draws<<",\"revisitedEdges\":"<<revisited_edges<<"}\n";
}
void verify_mirror(){
  for(const auto& [rows,columns]:std::array<std::pair<int,int>,2>{{{6,7},{7,6}}}){
    State board{0,0,static_cast<std::uint8_t>(rows),static_cast<std::uint8_t>(columns),true};
    for(int c=0;c<columns;++c)for(int r=0;r<rows;++r){
      Mask source=bit(board,c,r);
      Mask expected=bit(board,columns-1-c,r);
      if(mirror_mask(source,rows,columns)!=expected)
        throw std::runtime_error("Column-group mirror failed a basis-bit equivalence check.");
    }
    Mask used=0;
    for(int c=0;c<columns;++c)for(int r=0;r<rows;++r)used|=bit(board,c,r);
    if(mirror_mask(used,rows,columns)!=used)
      throw std::runtime_error("Column-group mirror failed the full-board equivalence check.");
    State sample{0,0,static_cast<std::uint8_t>(rows),static_cast<std::uint8_t>(columns),false};
    for(int c=0;c<columns;++c)for(int r=0;r<rows;++r){
      if((c+r)%3==0)sample.mover|=bit(sample,c,r);
      else if((2*c+r)%4==0)sample.opponent|=bit(sample,c,r);
    }
    State restored=mirror(mirror(sample));
    if(restored.mover!=sample.mover||restored.opponent!=sample.opponent
        ||restored.rows!=sample.rows||restored.columns!=sample.columns
        ||restored.ai_turn!=sample.ai_turn)
      throw std::runtime_error("Column-group mirror is not an involution.");
  }
}
void verify(){
  verify_mirror();
  for(std::uint8_t role:{std::uint8_t{1},std::uint8_t{2}}){
    std::string base="/tmp/c4-prefix-"+role_name(role);std::string p=base+".policy",f=base+".frontier",p2=base+"-6.policy",f2=base+"-6.frontier";
    run_segment({State{0,0,6,7,role==1}},role,0,4,1'000'000,p,f);
    FrontierInput input=read_frontier(f);run_segment(input.states,role,4,6,2'000'000,p2,f2);
    FrontierInput output=read_frontier(f2);if(output.boundary!=6||output.role!=role||output.states.empty())throw std::runtime_error("Layered verification failed.");
    std::remove(p.c_str());std::remove(f.c_str());std::remove(p2.c_str());std::remove(f2.c_str());
  }
}
} // namespace prefix
int main(int argc,char**argv){try{if(argc<2)throw std::runtime_error("Usage: perfect-chaos-prefix <verify|generate|extend|slice> ...");std::string c=argv[1];if(c=="verify")prefix::verify();else if(c=="generate")prefix::command_generate(argc,argv);else if(c=="extend")prefix::command_extend(argc,argv);else if(c=="slice")prefix::command_slice(argc,argv);else throw std::runtime_error("Unknown command: "+c);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
