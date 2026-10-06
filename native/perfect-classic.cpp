// Solves classic positions exactly on boards up to 7x7 (`solve`) and checks
// the solver on small boards with known values (`verify`). The search itself
// is native/classic-exact.hpp.
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "classic-exact.hpp"

namespace {

using namespace connect4::classic;

struct Arguments {
  std::string command = "verify";
  int rows = 6;
  int columns = 7;
  int connect = 4;
  int tableBits = 22;
  std::uint64_t maximumNodes = 0;
  std::string sequence;
};

Arguments parseArguments(int argc, char** argv) {
  Arguments arguments;
  int index = 1;
  if (index < argc && std::string_view(argv[index]).starts_with("--") == false) {
    arguments.command = argv[index++];
  }
  while (index < argc) {
    const std::string name = argv[index++];
    if (index >= argc) throw std::range_error(name + " requires a value");
    const std::string value = argv[index++];
    if (name == "--rows") arguments.rows = parseInt(value, 1, 7, "rows");
    else if (name == "--columns") arguments.columns = parseInt(value, 1, 7, "columns");
    else if (name == "--connect") arguments.connect = parseInt(value, 1, 7, "connect");
    else if (name == "--table-bits") arguments.tableBits = parseInt(value, 8, 27, "table-bits");
    else if (name == "--maximum-nodes") arguments.maximumNodes = parseUint64(value, "maximum-nodes");
    else if (name == "--sequence") arguments.sequence = value;
    else throw std::range_error("unknown argument: " + name);
  }
  return arguments;
}

Position positionFromSequence(const Geometry& geometry, std::string_view sequence) {
  Position position;
  for (const char token : sequence) {
    if (token == ',' || token == ' ' || token == '-') continue;
    if (token < '1' || token > '7') throw std::range_error("sequence must contain 1-based columns");
    if (hasAlignment(geometry, position.current ^ position.mask)) {
      throw std::range_error("sequence continues after a terminal win");
    }
    const int column = token - '1';
    const std::uint64_t move = moveForColumn(geometry, position.mask, column);
    if (move == 0) throw std::range_error("sequence contains an illegal move");
    position = play(position, move);
  }
  return position;
}

void printResult(
    const Geometry& geometry,
    const Position& position,
    int column,
    int value,
    const ExactSolver& solver,
    std::int64_t elapsedMs) {
  std::cout << "{\"format\":\"connect4-classic-exact-result-v1\""
            << ",\"rows\":" << geometry.rows
            << ",\"columns\":" << geometry.columns
            << ",\"connect\":" << geometry.connect
            << ",\"moves\":" << position.moves
            << ",\"value\":" << value
            << ",\"column\":" << column
            << ",\"nodes\":" << solver.nodes
            << ",\"tableHits\":" << solver.table().hits
            << ",\"tableStores\":" << solver.table().stores
            << ",\"tableCollisions\":" << solver.table().collisions
            << ",\"cutoffs\":" << solver.cutoffs
            << ",\"elapsedMs\":" << elapsedMs << "}\n";
}

void solveOne(const Arguments& arguments) {
  const Geometry geometry(arguments.rows, arguments.columns, arguments.connect);
  const Position position = positionFromSequence(geometry, arguments.sequence);
  const auto start = std::chrono::steady_clock::now();
  ExactSolver solver(geometry, arguments.tableBits, arguments.maximumNodes);

  int column = -1;
  int value;
  if (hasAlignment(geometry, position.current ^ position.mask)) {
    value = LOSS;
  } else if (hasAlignment(geometry, position.current)) {
    throw std::range_error("position contains a win for the side to move");
  } else {
    const auto result = solver.root(position);
    column = result.first;
    value = result.second;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();
  printResult(geometry, position, column, value, solver, elapsed);
}

void verify() {
  struct Case { int rows; int columns; int connect; int expected; };
  const std::array<Case, 6> cases{{
      {2, 2, 2, WIN},
      {3, 3, 3, DRAW},
      {4, 4, 3, WIN},
      {4, 4, 4, DRAW},
      {4, 5, 4, DRAW},
      {4, 6, 4, LOSS},
  }};
  for (const Case& test : cases) {
    const Geometry geometry(test.rows, test.columns, test.connect);
    ExactSolver solver(geometry, 20, 100'000'000);
    const auto result = solver.root(Position{});
    if (result.second != test.expected) {
      throw std::runtime_error("verification outcome mismatch");
    }
    std::cout << "{\"format\":\"connect4-classic-exact-verification-v1\""
              << ",\"rows\":" << test.rows
              << ",\"columns\":" << test.columns
              << ",\"connect\":" << test.connect
              << ",\"value\":" << result.second
              << ",\"column\":" << result.first
              << ",\"nodes\":" << solver.nodes << "}\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parseArguments(argc, argv);
    if (arguments.command == "verify") verify();
    else if (arguments.command == "solve") solveOne(arguments);
    else throw std::range_error("unknown command: " + arguments.command);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
