# python main.py --config ./config/config_baseline.json --suffix "beseline" --wandb_mode "online" >> baseline.txt
# python main.py --config ./config/config_baseline.json --suffix "baseline_with_model.eval_for_train_dice" --wandb_mode "online" >>baseline_correct_order.txt


# python main.py --config ./config/config_baseline.json --suffix "baseline:logits_for_loss+correct_order_for_sdc" --wandb_mode "online"  >> baseline:logits_for_loss+correct_order_for_sdc.txt
# python main.py --suffix "previous_commit" --config ./config/config_baseline.json --step "base_model" --wandb_mode "online" >> ./train_record/previous_coomit_baseline.txt

python main.py --step "feature_segmentor" --config ./config/feature_seg.json --suffix "old_commit" --wandb_mode "online" >>./train_record/old_commit_feature_seg.txt