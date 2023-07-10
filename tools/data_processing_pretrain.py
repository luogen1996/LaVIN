import json

#instruction:
#answer:
#input:
#options:
#qid:
#image
data=json.load(open('../../data/gcc.json'))
print(data)
#
#
# all_data=[]
#
# with open('../../data/chat.json') as f:
#     chat_data=json.load(f)
# for i,item in enumerate(chat_data):
#     data={}
#     data['instruction'] = 'Instruction: '+item['conversations'][0]['value'].replace('<image>\n','').replace('\n<image>','')+'\n'+ \
#                           'Response: '
#     data['instruction'] = data['instruction'].replace("  ", " ").strip()
#     data['answer'] = item['conversations'][1]['value']
#     data['image'] = item['image']
#     data['image_source']='gcc'
#     data['options'] = None
#     data['qid']='cc_'+str(i)
#     all_data.append(data)
#
#
#
#
# full_data={}
# full_data['all']=all_data
# with open('../../data/gcc.json','w') as f:
#     json.dump(full_data,f)
