# nohup bash finetuning2o.sh config=configs/conll05.crf2o.srl.bert.ini    path=exp/conll05.crf2o.srl.bert  bert=bert-large-cased   train=data/srl/conll05/train.conllu dev=data/srl/conll05/dev.conllu test=data/srl/conll05/test.conllu ood=data/srl/conll05/brown.conllu devices=4,5,6,7 > log.conll05.crf2o.bert    2>&1 &
# nohup bash finetuning2o.sh config=configs/conll05.crf2o.srl.roberta.ini path=exp/conll05.crf2o.srl.roberta bert=roberta-large    train=data/srl/conll05/train.conllu dev=data/srl/conll05/dev.conllu test=data/srl/conll05/test.conllu ood=data/srl/conll05/brown.conllu devices=4,5,6,7 > log.conll05.crf2o.roberta 2>&1 &
# nohup bash finetuning2o.sh config=configs/conll12.crf2o.srl.bert.ini    path=exp/conll12.crf2o.srl.bert    bert=bert-large-cased train=data/srl/conll12/train.conllu dev=data/srl/conll12/dev.conllu test=data/srl/conll12/test.conllu ood=data/srl/conll12/test.conllu  devices=4,5,6,7 > log.conll12.crf2o.bert 2>&1 &
# nohup bash finetuning2o.sh config=configs/conll12.crf2o.srl.roberta.ini path=exp/conll12.crf2o.srl.roberta bert=roberta-large    train=data/srl/conll12/train.conllu dev=data/srl/conll12/dev.conllu test=data/srl/conll12/test.conllu ood=data/srl/conll12/test.conllu  devices=4,5,6,7 > log.conll12.crf2o.roberta 2>&1 &
args=$@
for arg in $args; do
    eval "$arg"
done

DATA=~/.cache/supar/data

echo "config:  ${config:=config.ini}"
echo "path:    ${path:=exp/conll05.crf2o.srl.bert}"
echo "train:   ${train:=$DATA/srl/conll05/train.conllu}"
echo "dev:     ${dev:=$DATA/srl/conll05/dev.conllu}"
echo "test:    ${test:=$DATA/srl/conll05/test.conllu}"
echo "ood:     ${ood:=$DATA/srl/conll05/brown.conllu}"
echo "bert:    ${config:=bert-large-cased}"
echo "batch:   ${batch:=1000}"
echo "dropout: ${dropout:=0.1}"
echo "epochs:  ${epochs:=20}"
echo "rate:    ${rate:=20}"
echo "nu:      ${nu:=0.9}"
echo "eps:     ${eps:=1e-12}"
echo "devices: ${devices:=4,5,6,7}"

IFS=',' read -r -a device_arr <<< "$devices"
n_devices=${#device_arr[@]}

export TOKENIZERS_PARALLELISM=true
train() {
    # run processes and store pids in array
    for seed in {0..3}; do
        device_id=$(($seed % $n_devices))
        if [ ${#pids[@]} -gt $device_id ]; then
            echo "wait ${pids[$device_id]} to be done"
            wait ${pids[$device_id]}
        fi
        printf "nohup python -u crf2o.py train -b -c $config -s $seed -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed  --batch-size=$batch --mlp-dropout=$dropout --epochs=$epochs --lr-rate=$rate --train $train --dev $dev --test $test --encoder bert --bert $bert 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.train.log.verbose &\n\n"
        nohup python -u crf2o.py train -b -c $config -s $seed -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --batch-size=$batch --mlp-dropout=$dropout --epochs=$epochs --lr-rate=$rate --train $train --dev $dev --test $test --encoder bert --bert $bert 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.train.log.verbose &
        pids[${i}]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
}

evaluate() {
    # run processes and store pids in array
    for dataset in dev test ood; do
        for seed in {0..3}; do
            echo $seed $dataset ${!dataset}
            device_id=$(($seed % $n_devices))
            if [ ${#pids[@]} -gt $device_id ]; then
                echo "wait ${pids[$device_id]} to be done"
                wait ${pids[$device_id]}
            fi
            printf "nohup python -u crf2o.py evaluate -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.evaluate.log.verbose &\n\n"
            nohup python -u crf2o.py evaluate -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} >$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.evaluate.log.verbose &
            pids[${i}]=$!
            echo ${pids[*]}
        done
    done
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    for dataset in dev test ood; do
        for seed in {0..3}; do
            device_id=$(($seed % $n_devices))
            if [ ${#pids[@]} -gt $device_id ]; then
                echo "wait ${pids[$device_id]} to be done"
                wait ${pids[$device_id]}
            fi
            printf "nohup python -u crf2o.py evaluate -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --prd 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.prd.evaluate.log.verbose &\n\n"
            nohup python -u crf2o.py evaluate -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --prd >$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.prd.evaluate.log.verbose &
            pids[${i}]=$!
            echo ${pids[*]}
        done
    done
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
}

predict() {
    run processes and store pids in array
    for dataset in dev test ood; do
        for seed in {0..3}; do
            echo $seed $dataset ${!dataset}
            device_id=$(($seed % $n_devices))
            if [ ${#pids[@]} -gt $device_id ]; then
                echo "wait ${pids[$device_id]} to be done"
                wait ${pids[$device_id]}
            fi
            printf "nohup python -u crf2o.py predict -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --pred $path/$dataset.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.pred.conllu 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.predict.log.verbose &\n\n"
            nohup python -u crf2o.py predict -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --pred $path/$dataset.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.pred.conllu >$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.predict.log.verbose &
            pids[${i}]=$!
            echo ${pids[*]}
        done
    done
    for dataset in dev test ood; do
        for seed in {0..3}; do
            device_id=$(($seed % $n_devices))
            if [ ${#pids[@]} -gt $device_id ]; then
                echo "wait ${pids[$device_id]} to be done"
                wait ${pids[$device_id]}
            fi
            printf "nohup python -u crf2o.py predict -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --pred $path/$dataset.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.pred.gold.conllu --prd 2>$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.prd.predict.log.verbose &\n\n"
            nohup python -u crf2o.py predict -c $config -d ${device_arr[$device_id]} -p $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed --bert=$bert --data ${!dataset} --pred $path/$dataset.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.pred.gold.conllu --prd >$path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.prd.predict.log.verbose &
            pids[${i}]=$!
            echo ${pids[*]}
        done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
}

avg(){
    printf "Current commits:\n$(git log -1 --oneline)\n3rd parties:\n"
    cd 3rdparty/parser/ && printf "parser\n$(git log -1 --oneline)\n" && cd ../..
    for seed in {0..3}; do
        line=$(tail -n 2 $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.evaluate.log.verbose | head -n 1)
        echo $line
        ps[$seed]=${line:0-26:5}
        rs[$seed]=${line:0-16:5}
        fs[$seed]=${line:0-6:5}
    done
    printf "Average P/R/F score:\n"
    echo ${ps[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    echo ${rs[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    echo ${fs[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    printf "\n\n"
    echo 'w/ gold predicate'
    for seed in {0..3}; do
        line=$(tail -n 2 $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$seed.$dataset.prd.evaluate.log.verbose | head -n 1)
        echo $line
        ps[$seed]=${line:0-26:5}
        rs[$seed]=${line:0-16:5}
        fs[$seed]=${line:0-6:5}
    done
    printf "Average P/R/F score:\n"
    echo ${ps[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    echo ${rs[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    echo ${fs[@]} | awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; printf("%.2f ", sum)}'
    printf "\n\n"
}

collect() {
    echo $path/model.batch$batch.dropout$dropout
    for dataset in dev test ood; do
        echo "All cmds for $dataset has been done!" | tee -a $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$dataset.avg.log
        avg $dataset | tee -a $path/model.batch$batch.dropout$dropout.epochs$epochs.rate$rate.$dataset.avg.log
    done
    printf "\n"
}

mkdir -p $path
train
evaluate
predict
collect
