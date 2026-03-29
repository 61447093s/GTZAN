#pip install -r requirements.txt
#python train.py \
#  --csv_path gtzan.csv \
#  --audio_root genres \
#  --out_dir checkpoints_cnn \
#  --batch_size 64 \
#  --epochs 100 \
#  --validate_every 10 \
#  --lr 1e-3 \
#  --weight_decay 1e-4\
#  --dropout 0.5 \
#  --hop_length 512 \
#  --target_sr 22050

pip install -r requirements.txt

for bs in 32 64; do
  for lr in 0.0001 0.0005 0.001; do
    for wd in 0.0001 0.001; do
      for dr in 0.3 0.5; do
        for hop in 256 512; do
          echo "Training with batch=$bs, lr=$lr, wd=$wd, dropout=$dr, hop=$hop"
          python train.py \
            --csv_path gtzan.csv \
            --audio_root genres \
            --out_dir results/bs${bs}_lr${lr}_wd${wd}_dr${dr}_hop${hop} \
            --batch_size $bs \
            --epochs 100 \
            --validate_every 10 \
            --lr $lr \
            --weight_decay $wd \
            --dropout $dr \
            --hop_length $hop \
            --target_sr 22050
        done
      done
    done
  done
done
