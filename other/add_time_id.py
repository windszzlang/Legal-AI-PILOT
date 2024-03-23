import json



# dataset = 'ECHR2023_ext:5:9000'
def add(dataset):
    tmp_id = 0
    for split in ['train', 'dev', 'test']:
        all_cases = []
        with open('../data/' + dataset + '/' + split + '.jsonl') as f:
            for line in f.readlines():
                tmp_case = json.loads(line)
                tmp_case['time_id'] = tmp_id
                all_cases.append(tmp_case)
                tmp_id += 1

        with open('../data/' + dataset + '/' + split + '.jsonl', 'w') as f:
            for tmp_case in all_cases:
                f.write(json.dumps(tmp_case) + '\n')

# add('ECHR2023_ext:3:3000')
# add('ECHR2023_ext:3:6000')
# add('ECHR2023_ext:3:9000')
# add('ECHR2023_ext:5:3000')
# add('ECHR2023_ext:5:9000')
# add('ECHR2023_ext:7:3000')
# add('ECHR2023_ext:7:6000')
# add('ECHR2023_ext:7:9000')


add('ECHR2023_ext')
# add('ECHR2023_ext:3:' + str(1e10))
# add('ECHR2023_ext:5:' + str(1e10))
# add('ECHR2023_ext:7:' + str(1e10))
