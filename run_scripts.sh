python scripts/reconstruction.py exp_path=/share/portal/pd337/xskill/experiment/pretrain/no_pairing_singlehand_2024-05-31_12-03-03/ ckpt=27 ot_lookup=True tcc_lookup=True human_type=singlehand_segments_paired_sample num_chops=4

python scripts/label_sim_kitchen_dataset.py label_artificial=True artificial_type=tcc exp_path=/share/portal/pd337/xskill/experiment/pretrain/no_pairing_singlehand_2024-05-31_12-03-03/ ckpt=27 human_type=singlehand_segments_paired_sample skip_robot=True

python scripts/label_sim_kitchen_dataset.py label_artificial=True artificial_type=ot exp_path=/share/portal/pd337/xskill/experiment/pretrain/no_pairing_singlehand_2024-05-31_12-03-03/ ckpt=27 human_type=singlehand_segments_paired_sample skip_robot=True