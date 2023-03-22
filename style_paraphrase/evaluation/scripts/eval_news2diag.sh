#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

declare -a arr=(0.0 0.6 0.9)

gtype="nucleus_paraphrase"
split="test"

for top_p in "${arr[@]}"
do
    printf "\n-------------------------------------------\n"
    printf "Mode $gtype  --- "
    printf "top-p ${top_p}, split ${split}"
    printf "\n-------------------------------------------\n\n"

    # mkdir -p $1/eval_${gtype}_${top_p}

    # path0=$1/eval_${gtype}_${top_p}/transfer_dialogue_${split}.txt
    base_path0=$1/eval_${gtype}_${top_p}


    # printf "\ntranslate news to dialogue\n"
    # python -m style_paraphrase.evaluation.scripts.style_transfer \
    #     --style_transfer_model $1 \
    #     --input_file datasets/dialouge_dataset/test_news_only.txt \
    #     --output_file transfer_dialogue_${split}.txt \
    #     --generation_mode $gtype \
    #     --detokenize \
    #     --post_detokenize \
    #     --paraphrase_model $2 \
    #     --top_p ${top_p}

    printf "\nRoBERTa ${split} classification\n\n"
    python style_paraphrase/evaluation/scripts/roberta_classify.py --input_file ${base_path0}/transfer_dialogue_${split}.txt --label_file datasets/dialouge_dataset/test_news_only_dialogue_label.label --model_dir style_paraphrase/style_classify/saved_models/save_0 --model_data_dir datasets/dialouge_dataset/-bin

    printf "\nRoBERTa acceptability classification\n\n"
    python style_paraphrase/evaluation/scripts/acceptability.py --input_file ${base_path0}/transfer_dialogue_${split}.txt

    printf "\nParaphrase scores --- generated vs inputs..\n\n"
    python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${base_path0}/transfer_dialogue_${split}.txt --reference_strs reference --reference_paths datasets/dialouge_dataset/test_news_only.txt --output_path ${base_path0}/generated_vs_inputs.txt

    # printf "\nParaphrase scores --- generated vs gold..\n\n"
    # python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${base_path0}/all_${split}_generated.txt --reference_strs reference --reference_paths ${base_path0}/all_${split}_gold.txt --output_path ${base_path0}/generated_vs_gold.txt --store_scores

    # printf "\n final normalized scores vs gold..\n\n"
    # python style_paraphrase/evaluation/scripts/micro_eval.py --classifier_file ${base_path0}/all_${split}_generated.txt.roberta_labels --paraphrase_file ${base_path0}/all_${split}_generated.txt.pp_scores --generated_file ${base_path0}/all_${split}_generated.txt --acceptability_file ${base_path0}/all_${split}_generated.txt.acceptability_labels

done