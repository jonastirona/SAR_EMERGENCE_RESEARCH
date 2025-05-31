# AR Emergence code


### How to run

- Log into NAS and run the below command
    
    ```bash
    conda activate heliofm  

    # for transformer
    python train_w_stats.py 12 4 110 3 64 1000 0.01 st_transformer
    # for lstm
    python train_w_stats.py 12 4 110 3 64 1000 0.01 lstm
    ```

- For eval use the `shell_scripts/eval11726.sh` or you can run the below scripts
    ```
    python3 eval11726.py \
        --model_name st_transformer \
        --pth_file results/st_transformer/t12_r4_i110_n3_h64_e1000_l0.01.pth \
        --save_path results/st_transformer/t12_r4_i110_n3_h64_e1000_l0.01_eval.pdf \

    python3 eval11726.py \
        --model_name lstm \
        --pth_file results/lstm/t12_r4_i110_n3_h64_e1000_l0.01.pth \
        --save_path results/lstm/t12_r4_i110_n3_h64_e1000_l0.01.pdf \
    ```

- For eval use the `shell_scripts/eval13165.sh` or you can run the below scripts
    ```
    python3 eval13165.py \
        --model_name st_transformer \
        --pth_file results/st_transformer/t12_r4_i110_n3_h64_e1000_l0.01.pth \
        --save_path results/st_transformer/eval13165_t12_r4_i110_n3_h64_e1000_l0.01_eval.pdf \

    python3 eval13165.py \
        --model_name lstm \
        --pth_file results/lstm/t12_r4_i110_n3_h64_e1000_l0.01.pth \
        --save_path results/lstm/eval13165_t12_r4_i110_n3_h64_e1000_l0.01.pdf \
    ```
