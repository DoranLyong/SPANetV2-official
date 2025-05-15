config_path=local_configs/fpn/SPANetV2/fpn_spanet_s18_hybrid_ade20k-40k.py
ckpt_path=results/fpn_spanetv2_s18_hybrid_ade20k_best.pth

dataDir=/workspace/dataset/ade20k/ADEChallengeData2016
device=cuda:0  # cpu or cuda:0



# == Inference demo 
input=demo/demo.png
output=demo/result.png 
resultDir=demo/outputs

python demo/image_demo.py $input $config_path $ckpt_path --device $device --out-file $output


# == Inference to dataset 
for filePath in "$dataDir"/*".jpg"
do 
    fileName="$(basename "$filePath" suffix)"
    #echo "$fileName"

    "$(python demo/image_demo.py $filePath $config_path $ckpt_path --device $device --out-file $resultDir/$fileName)"
done



