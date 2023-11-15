# python main.py --config ./config/refinment.json --step "adapt"  --seed 42  --wandb_mode "online" --site 1 --data "cc359" --suffix "constrained_lora_epoch_5_batch:32_lr:0.0001_seed:42_target:ge_15" --adapt "constrained_lora" >> ./training/constrained_lora_ge_15.txt
# python main.py --config ./config/refinment.json --step "adapt"  --seed 42  --wandb_mode "online" --site 3 --data "cc359" --suffix "constrained_lora_epoch_5_batch:32_lr:0.0001_seed:42_target:philips_15" --adapt "constrained_lora" >> ./training/constrained_lora_philips_15.txt
# python main.py --config ./config/refinment.json --step "adapt"  --seed 42  --wandb_mode "online" --site 4 --data "cc359" --suffix "constrained_lora_epoch_5_batch:32_lr:0.0001_seed:42_target:philips_3" --adapt "constrained_lora" >> ./training/constrained_lora_philips_3.txt
# python main.py --config ./config/refinment.json --step "adapt"  --seed 42  --wandb_mode "online" --site 5 --data "cc359" --suffix "constrained_lora_epoch_5_batch:32_lr:0.0001_seed:42_target:siemens_15" --adapt "constrained_lora" >> ./training/constrained_lora_siemens_15.txt
# python main.py --config ./config/refinment.json --step "adapt"  --seed 42  --wandb_mode "online" --site 6 --data "cc359" --suffix "constrained_lora_epoch_5_batch:32_lr:0.0001_seed:42_target:siemens_3" --adapt "constrained_lora" >> ./training/constrained_lora_siemens_3.txt


# python main.py --config ./config/mms_feature_seg.json --data "mms" --step "feature_segmentor"  --seed 42  --wandb_mode "online"  --site "B" --suffix "ce_with_weights_epoch:50_batch:20_lr:0.001" 


# python main.py --config ./config/mms_config_refinment.json --data "mms" --step "adapt"  --seed 42  --wandb_mode "disabled"  --site "A" --suffix "control_experimet_bn_in_efs_train_val_set_weight" --adapt "constrained_lora" 
# python main.py --config ./config/mms_test_config.json --data "mms" --step "test" --seed 42  --wandb_mode "online" --site "A" --suffix "test"  --adapt "full_lora" --test test




# python main.py --config ./config/feature_seg.json --data "cc359" --step "feature_segmentor"  --seed 42  --wandb_mode "online"  --site 2 --suffix "level:full_lora_epoch:20_batch:32_lr:0.001_seed:42"


#python main.py --config ./config/refinment.json --data "cc359" --step "adapt"  --seed 42  --wandb_mode "online"  --site 1 --suffix "level:full_lora_epoch:5_batch:32_lr:0.0001_seed_42_target:ge_15" --adapt "full_lora"
python main.py --config ./config/refinment.json --data "cc359" --step "adapt"  --seed 42  --wandb_mode "online"  --site 3 --suffix "level:full_lora_epoch:5_batch:32_lr:0.0001_seed_42_target:philips_15" --adapt "full_lora"
python main.py --config ./config/refinment.json --data "cc359" --step "adapt"  --seed 42  --wandb_mode "online"  --site 4 --suffix "level:full_lora_epoch:5_batch:32_lr:0.0001_seed_42_:target:philips_3" --adapt "full_lora"
python main.py --config ./config/refinment.json --data "cc359" --step "adapt"  --seed 42  --wandb_mode "online"  --site 5 --suffix "level:full_lora_epoch:5_batch:32_lr:0.0001_seed_42_:target:siemens_15" --adapt "full_lora"
python main.py --config ./config/refinment.json --data "cc359" --step "adapt"  --seed 42  --wandb_mode "online"  --site 6 --suffix "level:full_lora_epoch:5_batch:32_lr:0.0001_seed_42_:target:siemens_3" --adapt "full_lora"