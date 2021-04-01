#!/bin/bash

export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
export PATH=~/.openmpi_4.0.0/bin:$PATH
export LD_LIBRARY_PATH=~/.openmpi_4.0.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH+=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH+=/usr/local/cuda-10.0/extras/CUPTI/lib64/

total_num_gpus=4
cluster_topology=shark8:4

MODEL_BASE_DIR=./pre-trained/base/xlnet_cased_L-12_H-768_A-12
MODEL_LARGE_DIR=./pre-trained/large/xlnet_cased_L-24_H-1024_A-16
GLUE_DIR=/ssd_dataset/dataset/glue

PRE_TRAINED_DIR=$MODEL_LARGE_DIR
#PRE_TRAINED_DIR=$MODEL_BASE_DIR
OUTPUT_DIR=./proc_data
MODEL_DIR=./model_ckpt

source ~/tf1.12_py3/bin/activate

mpirun -np $total_num_gpus \
    -H $cluster_topology \
    --prefix ~/.openmpi_4.0.0 \
    -bind-to none -map-by slot \
    -x PATH -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ib0 \
    python run_classifier_hvd.py \
      --do_train=True \
      --do_eval=False \
      --task_name=sts-b \
      --data_dir=${GLUE_DIR}/STS-B \
      --output_dir=${OUTPUT_DIR}/sts-b \
      --model_dir=${MODEL_DIR}/sts-b \
      --uncased=False \
      --spiece_model_file=${PRE_TRAINED_DIR}/spiece.model \
      --model_config_path=${PRE_TRAINED_DIR}/xlnet_config.json \
      --init_checkpoint=${PRE_TRAINED_DIR}/xlnet_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=8 \
      --learning_rate=5e-5 \
      --train_steps=1200 \
      --warmup_steps=120 \
      --is_regression=True

#     --save_steps=600 \
