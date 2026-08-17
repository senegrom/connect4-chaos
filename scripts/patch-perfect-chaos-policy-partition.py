#!/usr/bin/env python3
"""Install the exact policy-root dependency partition command.

The command follows a committed policy through every legal opponent reply,
builds the closed policy graph, and reverse-propagates from newly rejected
frontier states.  Input roots are then split into byte-stable unaffected and
affected frontier files.  This is the sound primitive needed to repair only
the portion of a prefix certificate that depends on a new counterexample.
"""

from __future__ import annotations

import sys
from pathlib import Path

PARTITION = r'''void command_partition(int argc,char**argv){
  std::string input_path,policy_path,reference_path,reject_path,unaffected_path,affected_path;
  for(int i=2;i<argc;++i){std::string arg=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error(arg+" needs a value.");return std::string(argv[i]);};
    if(arg=="--input-frontier")input_path=value();else if(arg=="--policy")policy_path=value();else if(arg=="--reference-frontier")reference_path=value();else if(arg=="--reject-frontier")reject_path=value();else if(arg=="--unaffected")unaffected_path=value();else if(arg=="--affected")affected_path=value();else throw std::runtime_error("Unknown argument: "+arg);
  }
  if(input_path.empty()||policy_path.empty()||reference_path.empty()||reject_path.empty()||unaffected_path.empty()||affected_path.empty())throw std::runtime_error("partition requires input, policy, reference, rejection, unaffected, and affected paths.");
  FrontierInput input=read_frontier(input_path);PolicyInput policy=read_policy(policy_path);FrontierInput reference=read_frontier(reference_path);FrontierInput rejected=read_frontier(reject_path);
  if(input.role!=policy.role||input.role!=reference.role||input.role!=rejected.role||policy.boundary!=reference.boundary||policy.boundary!=rejected.boundary||input.boundary>=policy.boundary)throw std::runtime_error("Partition table metadata does not align.");

  KeySet reference_keys;reference_keys.reserve(reference.states.size());for(const State& state:reference.states)reference_keys.insert(key_of(state));
  KeySet rejected_keys;rejected_keys.reserve(rejected.states.size());for(const State& state:rejected.states)rejected_keys.insert(key_of(state));

  struct DependencyNode{State state;std::vector<std::uint32_t> predecessors;};
  std::vector<DependencyNode> nodes;nodes.reserve(input.states.size()+policy.actions.size()+reference.states.size());
  std::unordered_map<Key,std::uint32_t,KeyHash> ids;ids.reserve(input.states.size()+policy.actions.size()+reference.states.size());
  std::queue<std::uint32_t> pending;std::vector<std::uint32_t> roots;roots.reserve(input.states.size());
  auto intern=[&](const State& raw){Canonical canonical=canonicalize(raw);auto [found,inserted]=ids.emplace(canonical.key,static_cast<std::uint32_t>(nodes.size()));if(inserted){nodes.push_back({canonical.state,{}});pending.push(found->second);}return found->second;};
  for(const State& state:input.states)roots.push_back(intern(state));

  std::vector<std::uint32_t> rejected_boundary_nodes;rejected_boundary_nodes.reserve(rejected.states.size());
  std::uint64_t closure_edges=0,terminal_ai_wins=0,terminal_draws=0,revisited_edges=0;
  std::uint32_t ai_states=0,opponent_states=0,boundary_states=0;
  auto follow=[&](std::uint32_t parent,const State& state,const Transition& transition){
    if(transition.mover_result!=0){Terminal terminal=terminal_for_ai(state,transition.mover_result);if(terminal==Terminal::AiLoss)throw std::runtime_error("Partitioned policy reaches an AI loss.");if(terminal==Terminal::AiWin)++terminal_ai_wins;else if(terminal==Terminal::Draw)++terminal_draws;return;}
    Canonical child=canonicalize(transition.state);auto found=ids.find(child.key);std::uint32_t child_id;
    if(found==ids.end()){child_id=static_cast<std::uint32_t>(nodes.size());ids.emplace(child.key,child_id);nodes.push_back({child.state,{}});pending.push(child_id);}else{child_id=found->second;++revisited_edges;}
    nodes[child_id].predecessors.push_back(parent);++closure_edges;
  };

  while(!pending.empty()){
    const std::uint32_t id=pending.front();pending.pop();const State state=nodes[id].state;const std::uint8_t count=pieces(state);
    if(count==policy.boundary){
      const Key key=key_of(state);if(!reference_keys.contains(key))throw std::runtime_error("Partitioned policy reaches a frontier state outside the reference certificate.");
      ++boundary_states;if(rejected_keys.contains(key))rejected_boundary_nodes.push_back(id);continue;
    }
    if(count>policy.boundary)throw std::runtime_error("Partitioned policy crossed its target frontier.");
    if(state.ai_turn){
      ++ai_states;auto found=policy.actions.find(key_of(state));if(found==policy.actions.end())throw std::runtime_error("Partitioned policy is missing a reachable AI state.");
      follow(id,state,apply(state,found->second,4));
    }else{
      ++opponent_states;std::array<bool,4> seen_terminal{};std::vector<Key> seen_children;seen_children.reserve(10);
      for(Action action:legal(state)){
        Transition transition=apply(state,action,4);
        if(transition.mover_result!=0){Terminal terminal=terminal_for_ai(state,transition.mover_result);std::size_t index=static_cast<std::size_t>(terminal);if(seen_terminal[index])continue;seen_terminal[index]=true;follow(id,state,transition);continue;}
        Canonical child=canonicalize(transition.state);if(std::find(seen_children.begin(),seen_children.end(),child.key)!=seen_children.end())continue;seen_children.push_back(child.key);follow(id,state,transition);
      }
    }
  }

  std::vector<std::uint8_t> affected_node(nodes.size(),0);std::queue<std::uint32_t> reverse;
  for(std::uint32_t id:rejected_boundary_nodes)if(!affected_node[id]){affected_node[id]=1;reverse.push(id);}
  while(!reverse.empty()){
    std::uint32_t child=reverse.front();reverse.pop();
    for(std::uint32_t parent:nodes[child].predecessors)if(!affected_node[parent]){affected_node[parent]=1;reverse.push(parent);}
  }

  std::vector<State> unaffected,affected;unaffected.reserve(input.states.size());affected.reserve(input.states.size());
  for(std::size_t index=0;index<input.states.size();++index){
    if(affected_node[roots[index]])affected.push_back(input.states[index]);else unaffected.push_back(input.states[index]);
  }
  write_frontier_states(unaffected_path,std::move(unaffected),input.role,input.boundary);write_frontier_states(affected_path,std::move(affected),input.role,input.boundary);
  FrontierInput unaffected_output=read_frontier(unaffected_path);FrontierInput affected_output=read_frontier(affected_path);
  if(unaffected_output.states.size()+affected_output.states.size()!=input.states.size())throw std::runtime_error("Partition outputs do not cover the input roots.");
  std::cout<<"{\"format\":\"connect4-chaos-policy-root-partition-v1\",\"role\":\""<<role_name(input.role)<<"\",\"fromPieces\":"<<static_cast<int>(input.boundary)<<",\"targetPieces\":"<<static_cast<int>(policy.boundary)
    <<",\"inputRoots\":"<<input.states.size()<<",\"unaffectedRoots\":"<<unaffected_output.states.size()<<",\"affectedRoots\":"<<affected_output.states.size()<<",\"sourcePolicyEntries\":"<<policy.actions.size()<<",\"sourceFrontierStates\":"<<reference.states.size()
    <<",\"rejectedTableStates\":"<<rejected.states.size()<<",\"rejectedBoundaryStatesReached\":"<<rejected_boundary_nodes.size()<<",\"closureStates\":"<<nodes.size()<<",\"closureEdges\":"<<closure_edges<<",\"aiStates\":"<<ai_states<<",\"opponentStates\":"<<opponent_states
    <<",\"boundaryStates\":"<<boundary_states<<",\"terminalAiWins\":"<<terminal_ai_wins<<",\"terminalDraws\":"<<terminal_draws<<",\"revisitedEdges\":"<<revisited_edges<<"}\n";
}
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-policy-partition.py SOURCE")
    path = Path(sys.argv[1])
    source = path.read_text()
    if "void command_partition(int argc,char**argv)" in source:
        print("Exact policy-root partition command is already installed.")
        return
    source = replace_once(
        source,
        "void command_slice(int argc,char**argv){",
        PARTITION + "void command_slice(int argc,char**argv){",
        "partition insertion",
    )
    source = replace_once(
        source,
        'Usage: perfect-chaos-prefix <verify|generate|extend|slice> ...',
        'Usage: perfect-chaos-prefix <verify|generate|extend|partition|slice> ...',
        "usage",
    )
    source = replace_once(
        source,
        'else if(c=="extend")prefix::command_extend(argc,argv);else if(c=="slice")prefix::command_slice(argc,argv);',
        'else if(c=="extend")prefix::command_extend(argc,argv);else if(c=="partition")prefix::command_partition(argc,argv);else if(c=="slice")prefix::command_slice(argc,argv);',
        "dispatch",
    )
    path.write_text(source)


if __name__ == "__main__":
    main()
