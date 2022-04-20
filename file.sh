#!/bin/bash

path1=E:'/'pycharm_projects'/'phase1'/'VidMain.py #the path of the pyhton script in case of depugging
path2=E:'\'pycharm_projects'\'phase1'\'debug.py#the path of the pyhton script in case of not depugging 
output=E:'\'pycharm_projects'\'phase1'\'output_video.mp4 #the path of the output files
inputfile=E:'\'pycharm_projects'\'phase1'\'challenge_video.mp4 #the path of the input video



while getopts d: flag
do
    case "${flag}" in
        d) depug=${OPTARG};;
    esac
done


#
case "$depug" in
    #case 1
    1) python $path1 $inputfile $output;;
      
    #case 0
    0) python $path2 $inputfile $output;;
      
	#default
	*) echo "Please ";;
esac


