#!/bin/sh -e

# Script to run Hyperscan testcase measurements. Pass in the path to your
# `hscollider` binary as an argument.
# hscollider -e [规则文件] -c [样本文件] -Z [对齐字节] -T [线程数目] -vv
if [ $# -ne 1 ]; then
    echo "Usage: ./run_bench.sh <hsbench binary>"
    exit 1
fi

HSCOLLIDER_BIN=$1

if [ ! -e ${HSCOLLIDER_BIN} ]; then
    echo "Can't find hscollider binary: ${HSCOLLIDER_BIN}"
    exit 1
fi

echo "\n*** hscollider test\n"
currpath=`pwd`
corpus="test_cases/corpora"
pcre="test_cases/pcre"

for element in `ls ${corpus}`
    do  
        dir_or_file="/"$element
        echo $element
        if [ -d $dir_or_file ]
        then 
            echo "error"
        else
            pcrefile=${currpath}"/"${pcre}$dir_or_file
            corpusfile=${currpath}"/"${corpus}$dir_or_file
            echo $pcrefile
            echo $corpusfile
            taskset 1 ${HSCOLLIDER_BIN} -e $pcrefile -c $corpusfile -Z 0 -T 10 &
        fi  
    done



