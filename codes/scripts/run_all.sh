#!/usr/bin/env bash
# **********************************************************************
# * Description   : run all experiments
# * Last change   : 13:57:23 2019-10-04
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

echo -e "Script: run for all experiments..."

CURRENT_DIR=`dirname "$0"`
CURRENT_DIR=`cd $CURRENT_DIR; pwd`
MAIN_DIR=${CURRENT_DIR}/../
RESULT_DIR=${CURRENT_DIR}/../result
cd $MAIN_DIR

[ ! -d "$RESULT_DIR" ] && mkdir $RESULT_DIR

# one hidden layer, relu motivation, euclidean loss
exp_1()
{
    echo -e "\t*exp_1"
    EXP_DIR=${RESULT_DIR}/exp_1
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_1 \
        > $LOG_PATH
}

# one hidden layer, sigmoid motivation, euclidean loss
exp_2()
{
    echo -e "\t*exp_2"
    EXP_DIR=${RESULT_DIR}/exp_2
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_2 -arch "Lin-784-10 Sigm" \
        > $LOG_PATH
}

# one hidden layer, relu motivation, softmax loss
exp_3()
{
    echo -e "\t*exp_3"
    EXP_DIR=${RESULT_DIR}/exp_3
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_3 -loss Softmax \
        > $LOG_PATH
}

# one hidden layer, sigmoid motivation, softmax loss
exp_4()
{
    echo -e "\t*exp_4"
    EXP_DIR=${RESULT_DIR}/exp_4
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_4 -loss Softmax -arch "Lin-784-10 Sigm" \
        > $LOG_PATH
}

# two hidden layer, relu motivation, euclidean loss
exp_5()
{
    echo -e "\t*exp_5"
    EXP_DIR=${RESULT_DIR}/exp_5
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_5 -arch "Lin-784-10 Relu Lin-10-10 Relu" -config "learning_rate:0.1 weight_decay:0.0001 momentum:0.1" \
        > $LOG_PATH
}

# two hidden layer, sigmoid motivation, euclidean loss
exp_6()
{
    echo -e "\t*exp_6"
    EXP_DIR=${RESULT_DIR}/exp_6
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_6 -arch "Lin-784-10 Sigm Lin-10-10 Sigm" -config "learning_rate:0.1 weight_decay:0.0001 momentum:0.1" \
        > $LOG_PATH
}

# two hidden layer, relu motivation, softmax loss
exp_7()
{
    echo -e "\t*exp_7"
    EXP_DIR=${RESULT_DIR}/exp_7
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_7 -arch "Lin-784-10 Relu Lin-10-10 Relu" \
        -config "learning_rate:0.1 weight_decay:0.0001 momentum:0.1" -loss Softmax \
        > $LOG_PATH
}

# two hidden layer, sigmoid motivation, softmax loss
exp_8()
{
    echo -e "\t*exp_8"
    EXP_DIR=${RESULT_DIR}/exp_8
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_8 -arch "Lin-784-10 Sigm Lin-10-10 Sigm" \
        -config "learning_rate:0.1 weight_decay:0.0001 momentum:0.1" -loss Softmax \
        > $LOG_PATH
}

# two hidden layer, relu and sigmoid motivation, softmax loss
exp_9()
{
    echo -e "\t*exp_9"
    EXP_DIR=${RESULT_DIR}/exp_9
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_9 -arch "Lin-784-10 Relu Lin-10-10 Sigm" \
        -config "learning_rate:0.1 weight_decay:0.0001 momentum:0.1" -loss Softmax \
        > $LOG_PATH
}

# three hidden layer, relu, relu and sigmoid motivation, softmax loss
exp_10()
{
    echo -e "\t*exp_10"
    EXP_DIR=${RESULT_DIR}/exp_10
    LOG_PATH=${EXP_DIR}/runtime.log
    [ ! -d "$EXP_DIR" ] && mkdir $EXP_DIR

    ./run.py -name exp_10 -arch "Lin-784-10 Relu Lin-10-10 Relu Lin-10-10 Sigm" \
        -config "learning_rate:1 weight_decay:0.00001 momentum:0.1" -loss Softmax \
        > $LOG_PATH
}

exp_1
exp_2
exp_3
exp_4
exp_5
exp_6
exp_7
exp_8
exp_9
exp_10
