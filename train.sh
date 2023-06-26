# python main.py --config ./config/config_baseline.json --suffix "beseline" --wandb_mode "online" >> baseline.txt
# python main.py --config ./config/config_baseline.json --suffix "baseline_with_model.eval_for_train_dice" --wandb_mode "online" >>baseline_correct_order.txt
# python main.py --config ./config/config_baseline.json --suffix "baseline:logits_for_loss+correct_order_for_sdc" --wandb_mode "online"  >> baseline:logits_for_loss+correct_order_for_sdc.txt

# python main.py --config ./config/config_baseline.json --suffix "old_commit:manual_slicing_epoch:100_batchsize:32" --step "base_model" --wandb_mode "online" >> ./train_record/old_commit:manual_slicing_epoch:100_batchsize:32.txt
# python main.py --suffix "old_commit:manual_slicing_epoch:100_batchsize:32"  --config ./config/refinment.json --step "feature_segmentor" --wandb_mode "online" >> ./train_record/head_old_commit:manual_slicing_epoch:100_batchsize:32.txt

# python main.py --suffix "on_ge_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 12 --site 1 >> ./train_record/on_ge_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset.txt
# python main.py --suffix "on_philips_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 12 --site 3 >> ./train_record/on_philips_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset.txt
# python main.py --suffix "on_philips_3:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 12 --site 4 >> ./train_record/on_philips_3:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset.txt
# python main.py --suffix "on_siemens_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 12 --site 5 >> ./train_record/on_siemens_15:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset.txt
# python main.py --suffix "on_siemens_3:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 12 --site 6 >> ./train_record/on_siemens_3:seed_12_epoch:100_batch_32_init_sdc_val_modes:trainset_valset.txt



# python main.py --config ./config/config_baseline.json --seed 12345 --suffix "seed_12345_manual_slicing_epoch:100_batch_32" --step "base_model" --wandb_mode "online" --site 2 >> ./train_record/base_model:seed_12345_manual_slicing_epoch:100_batch_32.txt



# python main.py --config ./config/refinment.json --seed 12345 --suffix "seed_12345_manual_slicing_epoch:100_batch_32" --step "feature_segmentor" --wandb_mode "online" --site 2 >>  ./train_record/head:seed_12345_manual_slicing_epoch:100_batch_32.txt


python main.py --suffix "ge_15_seed_123_manual_slicing_epoch:20_batch_32"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 1
python main.py --suffix "philips_15_seed_123_manual_slicing_epoch:20_batch_32"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 3
python main.py --suffix "philips_3_seed_123_manual_slicing_epoch:20_batch_32"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 4
python main.py --suffix "siemens_15_seed_123_manual_slicing_epoch:20_batch_32"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 5
python main.py --suffix "siemens_3_seed_123_manual_slicing_epoch:20_batch_32"  --config ./config/refinment.json --step "refine" --wandb_mode "online" --seed 123 --site 6


# python main.py --config ./config/refinment.json --seed 123 --suffix "seed_123_manual_slicing_epoch:20_batch_32" --step "feature_segmentor" --wandb_mode "online" --site 2 >>  ./train_record/seed_123_manual_slicing_epoch:20_batch_32.txt
