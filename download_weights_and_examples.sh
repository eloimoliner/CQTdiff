#!/bin/bash

wget https://github.com/eloimoliner/CQTdiff/releases/download/weights_and_examples/cqt_weights.pt
mkdir experiments
mkdir experiments/cqt
mv cqt_weights.pt  experiments/cqt/

wget https://github.com/eloimoliner/CQTdiff/releases/download/weights_and_examples/examples.zip
unzip examples.zip .
