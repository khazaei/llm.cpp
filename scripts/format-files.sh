#!/bin/bash

find ./src -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i
find ./test -iname '*.h' -o -iname '*.cpp' | xargs clang-format -i

# to run on only the file in the commit list.
#for FILE in $(git diff --cached --name-only)
#do
#  if [[ ${FILE} == *.cpp ]] || [[ ${FILE} == *.hpp ]] || [[ ${FILE} == *.h ]];then
#    clang-format -i $FILE
#  fi
#done