#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

#-- cfg 
config_path=local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py
ckpt_path=results/cascade-mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth

dataDir=/workspace/dataset/ade20k/ADEChallengeData2016
device=cuda:0  # cpu or cuda:0



# == Inference demo 
input=demo/demo.jpg
output=demo/result_SPANetV2
resultDir=demo/outputs

python demo/image_demo.py $input $config_path --weights $ckpt_path --device $device --out-dir $output


# == Inference to dataset 
#for filePath in "$dataDir"/*".jpg"
#do 
#    fileName="$(basename "$filePath" suffix)"
#    #echo "$fileName"
#
#    "$(python demo/image_demo.py $filePath $config_path $ckpt_path --device $device --out-file $resultDir/$fileName)"
#done



