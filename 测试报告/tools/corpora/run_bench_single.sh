#!/bin/sh -e

# Script to run Hyperscan performance measurements. Pass in the path to your
# `hsbench` binary as an argument.

if [ $# -ne 1 ]; then
    echo "Usage: ./run_bench.sh <hsbench binary>"
    exit 1
fi

HSBENCH_BIN=$1

if [ ! -e ${HSBENCH_BIN} ]; then
    echo "Can't find hsbench binary: ${HSBENCH_BIN}"
    exit 1
fi

echo "\n*** Snort literals against alexa200 text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/alexa200.db -N -T 1

echo "\n*** Snort literals against Gutenberg text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/gutenberg.db -N -T 1
echo

echo "\n*** Snort literals against news text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/news.db -N -T 1
echo

echo "\n*** Snort PCREs against alexa200 text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/alexa200.db -N -T 1
echo

echo "\n*** Snort PCREs against Gutenberg text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/gutenberg.db -N -T 1
echo

echo "\n*** Snort PCREs against news text, block mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/news.db -N -T 1
echo

echo "\n*** Snort literals against HTTP traffic, alexa200 text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/alexa200.db -T 1
echo

echo "\n*** Snort literals against HTTP traffic, Gutenberg text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/gutenberg.db -T 1
echo

echo "\n*** Snort literals against HTTP traffic, news text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_literals -c corpora/news.db -T 1
echo

echo "\n*** Snort PCREs against HTTP traffic, alexa200 text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/alexa200.db -T 1
echo

echo "\n*** Snort PCREs against HTTP traffic, Gutenberg text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/gutenberg.db -T 1
echo

echo "\n*** Snort PCREs against HTTP traffic, news text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/snort_pcres -c corpora/news.db -T 1
echo

echo "\n*** Teakettle synthetic patterns against alexa200 text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/teakettle_2500 -c corpora/alexa200.db -T 1
echo

echo "\n*** Teakettle synthetic patterns against Gutenberg text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/teakettle_2500 -c corpora/gutenberg.db -T 1
echo

echo "\n*** Teakettle synthetic patterns against news text, streaming mode.\n"
taskset 1 ${HSBENCH_BIN} -e pcre/teakettle_2500 -c corpora/news.db -T 1
echo