#!/bin/bash -x

STAGE=${1:-dataset1}
LANGUAGE=${2:-python}

unzip data/$LANGUAGE-$STAGE -d data/repositories-$LANGUAGE-$STAGE

for zipfile in data/repositories-$LANGUAGE-$STAGE/*.zip; do
  unzip -o "$zipfile" -d "${zipfile%.zip}"
done
