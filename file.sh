#!/bin/bash




while getopts ":d:p:i:o" options
do
    case "${options}" in
        d) flag=${OPTARG};;
        p) path1=${OPTARG};;
        i) inputfile=${OPTARG};;
        o) output=${OPTARG};;

    esac
done


#
case "$flag" in
    #case 1
    1) python $path1 $inputfile $output;;
 
      
	#default
    *) echo "Please ";;
esac


