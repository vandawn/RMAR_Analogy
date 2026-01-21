for lr in 5e-5
do
for fusion_strat in "gate" # "gate" "concat"
do
for alpha in 0.43
do
for bsz in 32 
do
for num_neighbors in 6
do
for temp in 1.0
do

echo "Running fine-tune with LR: $lr, Fusion: $fusion_strat, Alpha: $alpha, Batch Size: $bsz, Num Neighbors: $num_neighbors, Temperature: $temp"

python main_struct.py \
    --gpus "0," \
    --max_epochs=30  \
    --num_workers=4 \
    --model_name_or_path bert-base-uncased \
    --visual_model_path openai/clip-vit-base-patch32 \
    --accumulate_grad_batches 1 \
    --data_class data.data_module_struct.KGC \
    --model_class models.model_struct.MKGformerKGC \
    --litmodel_class lit_models.transformer_struct.TransformerLitModel \
    --batch_size $bsz \
    --pretrain 0 \
    --bce 0 \
    --check_val_every_n_epoch 1 \
    --overwrite_cache \
    --data_dir dataset/MCNetAnalogy \
    --pretrain_path dataset/MCNetKG \
    --eval_batch_size 128 \
    --max_seq_length 128 \
    --lr $lr \
    --alpha $alpha \
    --fusion_strategy $fusion_strat \
    --checkpoint pretrain.ckpt \
    --memory_size 4096 \
    --momentum_m 0.999 \
    --diversity_threshold 0.95 \
    --num_neighbors $num_neighbors \
    --print_freq 500 \
    --temperature $temp \
    --structural_embedding_path KG_embedding.pt \
    --structural_data_dir dataset/MCNetKG \
    --use_structure 1 \
    --use_r_enhance 1 \
    --use_memory_bank 1

done
done
done
done
done
done
