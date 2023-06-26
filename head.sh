# python main.py --config ./config/refinment.json --seed 123 --suffix "seed_123_manual_slicing_epoch:100_batch_32" --step "feature_segmentor" --wandb_mode "online" --site 2 >>  ./train_record/head:seed_123_manual_slicing_epoch:100_batch_32.txt

python main.py --suffix "ge_15_seed_123_with_head_saved_at_best_epoch"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 1
python main.py --suffix "philips_15_seed_123_with_head_saved_at_best_epoch"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 3
python main.py --suffix "philips_3_seed_123_with_head_saved_at_best_epoch"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 4
python main.py --suffix "siemens_15_seed_123_with_head_saved_at_best_epoch"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 5
python main.py --suffix "siemens_3_seed_123_with_head_saved_at_best_epoch"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 6