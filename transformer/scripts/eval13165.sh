python3 eval13165.py \
    --model_name st_transformer \
    --pth_file results/st_transformer/t12_r4_i110_n3_h64_e1000_l0.01.pth \
    --save_path results/st_transformer/eval13165_t12_r4_i110_n3_h64_e1000_l0.01_eval.pdf \

python3 eval13165.py \
    --model_name lstm \
    --pth_file results/lstm/t12_r4_i110_n3_h64_e1000_l0.01.pth \
    --save_path results/lstm/eval13165_t12_r4_i110_n3_h64_e1000_l0.01.pdf \

