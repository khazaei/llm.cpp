#include <catch2/catch_session.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <string>

int main(int argc, char *argv[]) {
  Catch::Session session; // There must be exactly one instance

  // Build a new parser on top of Catch's
  using namespace Catch::Clara;
  auto projectDir = std::string{};
  auto cli =
      session.cli() // Get Catch's composite command line parser
      | Opt(projectDir, "Project directory")["--ProjDir"]("Project root directory path");

  // Now pass the new composite back to Catch, so it uses that
  session.cli(cli);

  // Let Catch (using Clara) parse the command line
  const auto returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) { // Indicates a command line error
    return returnCode;
  }

  // use the parsed arguments

  return session.run();
}
