import json

#instruction:
#answer:
#input:
#options:
#qid:
#image
all_data=[]
with open('../data/alpaca_data.json') as f:
    alpaca_data=json.load(f)
for i,item in enumerate(alpaca_data):
    data={}
    input=item['input']
    if len(input)==0:
        input=''
    data['instruction'] = 'Instruction: '+ item['instruction']+' '+input+'\n'+\
                           'Response: '
    data['instruction'] = data['instruction'].replace("  ", " ").strip()
    data['answer'] = item['output']
    data['image'] = None
    data['options'] = None
    data['image_source'] = None
    data['qid']='alpaca_'+str(i)
    all_data.append(data)


with open('../data/complex_reasoning_77k.json') as f:
    gpt4_data_0=json.load(f)
with open('../data/detail_23k.json') as f:
    gpt4_data_1=json.load(f)
with open('../data/conversation_58k.json') as f:
    gpt4_data_2=json.load(f)
gpt4_data=gpt4_data_0+gpt4_data_1
for i,item in enumerate(gpt4_data):
    data={}
    data['instruction'] = 'Instruction: '+item['conversations'][0]['value'].replace('<image>\n','').replace('\n<image>','')+'\n'+ \
                          'Response: '
    data['instruction'] = data['instruction'].replace("  ", " ").strip()
    data['answer'] = item['conversations'][1]['value']
    data['image'] = item['image']
    data['image_source']='mscoco'
    data['options'] = None
    data['qid']='gpt4_'+str(i)
    all_data.append(data)


for i,item in enumerate(gpt4_data_2):
    for j in range(0,len(item['conversations']),2):
        data={}
        data['instruction'] = 'Instruction: '+item['conversations'][j]['value'].replace('<image>\n','').replace('\n<image>','')+'\n'+ \
                              'Response: '
        data['instruction'] = data['instruction'].replace("  ", " ").strip()
        data['answer'] = item['conversations'][j+1]['value']
        data['image'] = item['image']
        data['image_source']='mscoco'
        data['options'] = None
        data['qid']='gpt4_2_'+str(i)+'_'+str(j)
        all_data.append(data)



full_data={}
full_data['all']=all_data
with open('../data/all_data.json','w') as f:
    json.dump(full_data,f)
