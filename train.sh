# python main.py --config ./config/config_baseline.json --suffix "beseline" --wandb_mode "online" >> baseline.txt
# python main.py --config ./config/config_baseline.json --suffix "baseline_with_model.eval_for_train_dice" --wandb_mode "online" >>baseline_correct_order.txt


python main.py --config ./config/config_baseline.json --suffix "baseline:logits_for_loss+correct_order_for_sdc" --wandb_mode "online"  >> baseline:logits_for_loss+correct_order_for_sdc.txt