import os

models = [
'maco',
]
cpts = "maco.pth"

gpus = [0] #3
time = '1'

'''CheXpert'''
holestr=list(('' for x in range(len(gpus))))
j=0
for i in range(len(models)):
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j])+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ ' --learning_rate 8e-4;')
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j])+';')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j])+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ " --data_volume '10' --num_steps 60000 --learning_rate 1e-4 --warmup_steps 1500;")
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j])+';')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j])+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ " --data_volume '100' --num_steps 200000 --learning_rate 1e-5 --warmup_steps 15000;") 
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j])+';')
    j+=1
print('\n')
for i in holestr:
    print('CheXpert')
    print(i+'\n')

# '''NIH'''
holestr=list(('' for x in range(len(gpus))))
j=0
for i in range(len(models)):
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+1)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ '  --learning_rate 8.5e-3; ')
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+1)+'; ')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+1)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ "  --data_volume '10' --num_steps 30000 --learning_rate 3e-3 --warmup_steps 500; ")
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+1)+ '; ')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+1)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ " --data_volume '100'  --num_steps 200000 --learning_rate 3e-3 --warmup_steps 5000;")
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+1)+ ';')
    j+=1
print('\n')
for i in holestr:
    print('NIH')
    print(i+'\n')
    
# '''RSNA'''
holestr=list(('' for x in range(len(gpus))))
j=0
for i in range(len(models)):
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+2)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ ' --learning_rate 6.5e-3;')
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+2)+';')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+2)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ "  --data_volume '10' --num_steps 10000 --learning_rate 6e-4 --warmup_steps 200;")
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+2)+';')
    holestr[j]+=('CUDA_VISIBLE_DEVICES='+str(gpus[j]+2)+' python train.py --pretrained_path '+ '/path/to/model/'+str(models[i])+'/'+cpts+ " --data_volume '100' --num_steps 50000 --learning_rate 5e-4 --warmup_steps 2000;")
    holestr[j]+=('python test.py --model '+str(models[i])+' --gpu '+str(gpus[j]+2)+';')
    j+=1
print('\n')
for i in holestr:
    print('RSNA')
    print(i+'\n')