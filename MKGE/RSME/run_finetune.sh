for lr in 5e-3
do
echo ${lr}
python learn.py \
    --dataset="MCNet" \
    --model="Analogy" \
    --batch_size=1000 \
    --learning_rate=${lr} \
    --max_epochs=300 \
    --finetune \
    --ckpt="/home/rwan551/code/MKG_Analogy-main/M-KGE/RSME/checkpoint/pt_best_model_analogy_MCNet_30.pth"
done