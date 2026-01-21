for lr in 1e-2
do
echo ${lr}
python learn.py \
    --dataset="MCNet" \
    --model="Analogy" \
    --batch_size=1000 \
    --learning_rate=${lr} \
    --max_epochs=30
done