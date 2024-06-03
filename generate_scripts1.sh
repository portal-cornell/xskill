export base_dev_dir="/share/portal/pd337/xskill/experiment/pretrain/"
export dataset="singlehand_segments_paired_sample"
export human_demo="singlehand"

export model="no_pairing_singlehand_2024-05-31_12-03-03"
export model_name="${base_dev_dir}${model}/"
echo python scripts/label_sim_kitchen_dataset.py exp_path=$model_name ckpt=27 human_type=${dataset} skip_human=True
echo
echo python scripts/chopped_segment_wise_dists.py exp_path=$model_name ckpt=27 num_chops=4 human_type=${dataset}
echo
echo python scripts/reconstruction.py exp_path=$model_name ckpt=27 ot_lookup=True tcc_lookup=True human_type=${dataset} num_chops=4
echo
echo python scripts/label_sim_kitchen_dataset.py label_artificial=True artificial_type=tcc exp_path=$model_name ckpt=27 human_type=${dataset} skip_robot=True
echo
echo python scripts/label_sim_kitchen_dataset.py label_artificial=True artificial_type=ot exp_path=$model_name ckpt=27 human_type=${dataset} skip_robot=True
echo
echo python scripts/label_sim_kitchen_dataset.py exp_path=$model_name ckpt=27 human_type=${human_demo} skip_robot=True