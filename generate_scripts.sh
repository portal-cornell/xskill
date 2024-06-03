export base_dev_dir="/share/portal/kk837/xskill/experiment/pretrain/"
export dataset="singlehand_segments_paired_sample"
export human_demo="singlehand"
export model="no_pairing_singlehand_2024-05-31_12-03-03"
export model_name="${base_dev_dir}${model}/"
export true_proto_dir="${model_name}/${human_demo}_encode_protos/ckpt_27"
export ot_proto_dir="${model_name}/${dataset}_generated_ot_encode_protos/ckpt_27"
export tcc_proto_dir="${model_name}/${dataset}_generated_tcc_encode_protos/ckpt_27"
echo python scripts/skill_transfer_composing.py policy_name=singlehand_pretrain_no_pairing_true_pairing pretrain_path=${model_name} pretrain_ckpt=27 human_type=${human_demo} eval_cfg.demo_type=${human_demo} dataset.paired_data=True dataset.paired_percent=0.5 dataset.paired_proto_dirs=${true_proto_dir}
echo
echo python scripts/skill_transfer_composing.py policy_name=singlehand_pretrain_no_pairing_ot_pairing pretrain_path=${model_name} pretrain_ckpt=27 human_type=${dataset} eval_cfg.demo_type=${human_demo} dataset.paired_data=True dataset.paired_percent=0.5 dataset.paired_proto_dirs=${ot_proto_dir}
echo
echo python scripts/skill_transfer_composing.py policy_name=singlehand_pretrain_no_pairing_tcc_pairing pretrain_path=${model_name} pretrain_ckpt=27 human_type=${dataset} eval_cfg.demo_type=${human_demo} dataset.paired_data=True dataset.paired_percent=0.5 dataset.paired_proto_dirs=${tcc_proto_dir}
echo
echo python scripts/skill_transfer_composing.py policy_name=singlehand_pretrain_no_pairing_no_pairing pretrain_path=${model_name} pretrain_ckpt=27 human_type=${dataset} eval_cfg.demo_type=${human_demo}