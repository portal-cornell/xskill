export folder_path="/share/portal/pd337/xskill/experiment/pretrain"

# export amt=100
# export pretrain_path="${folder_path}/TWOHAND_100_pairing_twohands_segments_2024-06-04_01-52-02"
# export ckpt_num=40

# export amt=75
# export pretrain_path="${folder_path}/TWOHAND_75_pairing_twohands_segments_2024-06-04_01-52-08"
# export ckpt_num=60

# export amt=50
# export pretrain_path="${folder_path}/TWOHAND_50_pairing_twohands_segments_2024-06-04_01-52-27"
# export ckpt_num=79

export amt=25
export pretrain_path="${folder_path}/TWOHAND_25_pairing_twohands_segments_2024-06-04_01-52-42"
export ckpt_num=79

# echo python scripts/label_sim_kitchen_dataset.py exp_path=${pretrain_path} ckpt=${ckpt_num} 
# echo python scripts/skill_transfer_composing.py policy_name=TWOHANDS_pretrain_${amt}_pairing_ot_pairing pretrain_path=${pretrain_path} pretrain_ckpt=${ckpt_num} human_type=TWOHANDS_${amt}_PAIRING_OT eval_cfg.demo_type=twohands dataset.paired_data=True dataset.paired_percent=0.5 

echo python scripts/skill_transfer_composing.py policy_name=TWOHANDS_pretrain_${amt}_pairing_no_pairing pretrain_path=${pretrain_path} pretrain_ckpt=${ckpt_num} eval_cfg.demo_type=twohands dataset.paired_data=False

