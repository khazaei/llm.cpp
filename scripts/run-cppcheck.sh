#!/bin/bash

cppcheck ./src ./test ./app --error-exitcode=1 --enable=all --inline-suppr --suppress=useStlAlgorithm --suppress=missingInclude --suppress=missingIncludeSystem
if [[ $? -ne 0 ]]; then
  exit 1
fi