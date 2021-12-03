# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
CONFIG=configs/dual-pixel
EXPERIMENT=dual-pixel-smoothness
# DATA_DIR=/home/akashsharma/Documents/datasets/dual-pixel-defocus/
# TRAIN_DIR=/home/akashsharma/tmp/dual-pixel/$EXPERIMENT

DATA_DIR=/data3/tkhurana/datasets/dual-pixel-defocus/
TRAIN_DIR=/data3/tkhurana/misc/dualpixel/results/$EXPERIMENT

python -m train \
--data_dir=$DATA_DIR \
--train_dir=$TRAIN_DIR \
--config=$CONFIG \
--logtostderr
