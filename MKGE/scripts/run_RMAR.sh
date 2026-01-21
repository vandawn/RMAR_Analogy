# choices=["Mformer_hd_mean", "Mformer_hd_graph", "Mformer_weight", "atten_weight", "learnable_weight"]
# DATA=DB15K;MKG-W;MKG-Y
DATA=$2
EMB_DIM=128
NUM_BATCH=1024
MARGIN=12
LR=2e-3
LRG=1e-3
NEG_NUM=32
EPOCH=200
NU=epoch
NOISE=1
POOL=1
DISTRIBUTION_FITTING=1
FUSION_TYPE="single"  # "full", "single"

# GAUSSIAN=1
# TAIL_WEIGHT=0
# CONTRASTIVE=0
# NOISE_TYPE="mixture_noise"  #"dual_noise", "single_noise_mu", "single_noise_var", "mixture_noise"


CUDA_VISIBLE_DEVICES=$1 python run_RMAR.py  -dataset=$DATA \
  -num_proj=$3 \
  -use_intermediate=$4 \
  -joint_way=$5 \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=$EPOCH \
  -dim=$EMB_DIM \
  -save=$DATA-$NUM_BATCH-$EMB_DIM-$NEG_NUM-$MARGIN-$LR-$2 \
  -neg_num=$NEG_NUM \
  -noise_update=$8 \
  -noise_ratio=$6 \
  -mask_ratio=$7 \
  -use_pool=$POOL \
  -add_noise=$NOISE \
  -learning_rate=$LR \
  -num_hidden_layers=$9 \
  -num_attention_heads=${10} \
  -distribution_fitting=$DISTRIBUTION_FITTING \
  -fusion_type=$FUSION_TYPE \
  -exp_id=$EMB_DIM-$NEG_NUM-$3_layer_$4_inter_$5_${11} \
  # -finetune=${11} \
  # -noise_type=$NOISE_TYPE \
  # -tail_weight=$TAIL_WEIGHT \
  # -contrastive=$CONTRASTIVE \
  # -gaussian=$GAUSSIAN \
  

  
