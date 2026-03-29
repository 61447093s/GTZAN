python predict.py \
  --csv_path gtzan.csv \
  --audio_root genres \
  --ckpt_path results/bs32_lr0.0001_wd0.0001_dr0.5_hop512/checkpoint_best.pt \
  --out_csv bs32_lr0.0001_wd0.0001_dr0.5_hop512.csv \
  --dropout 0.5 \
  --hop_length 512 \
  --target_sr 0
