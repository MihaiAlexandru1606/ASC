#!/usr/bin/env bash

function determine_size() {
    start_val=$2

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -ne 0 ] ; then
            break
        fi

        start_val=$(($start_val + 100000))
    done

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -eq 0 ]; then
            break
        fi

        start_val=$(($start_val - 10000))
    done

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -ne 0 ]; then
            break
        fi

        start_val=$(($start_val + 1000))
    done

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -eq 0 ]; then
            break
        fi

        start_val=$(($start_val - 100))
    done

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -ne 0 ]; then
            break
        fi

        start_val=$(($start_val + 10))
    done

    while true
    do
        ./$1 $start_val &> /dev/null
        if [ $? -eq 0 ]; then
            break
        fi

        start_val=$(($start_val - 1))
    done

    echo "${start_val}"
}

function build() {
    make -f Makefile.task1a build &> /dev/null
    if [ $? -ne 0 ]; then
	    echo "Makefile error A!"
	    exit 1
    fi

    make -f Makefile.task1b build &> /dev/null
    if [ $? -ne 0 ]; then
	    echo "Makefile error B!"
	    exit 1
    fi

    make -f Makefile.task1c build &> /dev/null
    if [ $? -ne 0 ]; then
	    echo "Makefile error C!"
	    exit 1
    fi
}

function clean() {
    make -f Makefile.task1a clean &> /dev/null
    make -f Makefile.task1b clean &> /dev/null
    make -f Makefile.task1c clean &> /dev/null
}

start=1000000
exe_taska=task1a
exe_taskb=task1b
exe_taskc=task1c

build

val_bss=$( determine_size ${exe_taska}  ${start})
val_stack=$( determine_size ${exe_taskb}  ${start})
val_heap=$( determine_size ${exe_taskc} ${start})

echo "Val bss: $val_bss"
echo "Val stack: $val_stack"
echo "Val heap: $val_heap"

clean