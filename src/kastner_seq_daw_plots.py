import re
import pandas as pd
mf_gt_mb_seqs = ['B', 'B2', 'a1', '2b', 'b2', '2B2', '1A1', '1a1', '2b2', 'a1A', 'b2B', '2A2', 'B2b', 'a1a', '1B1', '2a1', '1b2', 'b2b', 'a1B', 'a1b', 'b2A', 'b2a', '2B2B', '1A1A', 'B2B2', 'A1A1', '2B2b', '1A1a', '1a1A', '2b2B', 'B2b2', 'b2B2', 'a1A1', '2B2A', 'A1a1', 'A2B2', 'B1A1', '2b2b', 'B2A2', '1B1A', '2A2B', '1a1a', '1A1b', 'B1a1', 'b2b2', 'a1a1', '1A1B', 'B1B1', 'A1B1', '1B1a', '1a1B', '2a1A', 'A2b2', '2A2b', '1b2b', 'b1B1', '1b2B', '2B2a', 'A2A2', 'a1B1', '1B1B', '2b2A', '2a1a', '2A2a', 'a2A2', 'B2a1', '2A2A', '2b2a', 'A1b2', 'b2a1', 'b1b2', 'b2A2', 'a2a1', 'a1b2', '1B1b', 'B1b2', 'b2a2', '1b2a', '2a1b', '2a1B', '1b2A']

def process_choice_state_choice(c1, s, c2):
    # e.g. 1, 'A', 2 -> False, True, True (no stick, yes rew, common transition)
    stick = c1 == c2
    rew = s.isupper()
    common = (c1, s.lower()) in [(1, 'a'), (2, 'b')]
    return stick, rew, common

def kastner_seq_to_dict(seq):
    """e.g. seq='B2b2a1A2
    ->
    [{'stick': True, 'rew': False, 'common_transition': True, 'csc': '2b2'},
    {'stick': False, 'rew': False, 'common_transition': False, 'csc': '2a1'},
    {'stick': False, 'rew': True, 'common_transition': True, 'csc': '1A2'}]'
    """
    matches = re.finditer('(?=(\d[a-zA-Z]\d))', seq)
    choice_state_choices = [match.group(1) for match in matches]

    # e.g. 1, 'A', 2 (rew common switch)
    choice_state_choice_tuples = [(int(c1), s, int(c2)) for c1, s, c2 in choice_state_choices]
    
    def get_out(csc):
        c1, s, c2 = csc
        stick, rew, common_transition = process_choice_state_choice(c1, s, c2)
        return {'stick': stick, 'rew': rew, 'common_transition': common_transition}
    
    outs = list(map(get_out, choice_state_choice_tuples))
    outs = list(map(lambda out_dict, csc: {**out_dict, 'csc': csc}, outs, choice_state_choices))
    return outs
    

def process_episode_data_flat(seq):
    """e.g. seq is a 400 x 2 array of subjects data
    
    [  [1, 0], # choice1, _
       [3, 1], # state, rew
       [1, 0], # choice, _
       [2, 1],
       [1, 0],
       [2, 0], ...]
    
    returns
    [{'stick': True, 'rew': True, 'common_transition': True, 'csc': '2B2'},
    {'stick': True, 'rew': True, 'common_transition': False, 'csc': '2A2'},
    {'stick': False, 'rew': False, 'common_transition': False, 'csc': '2a1'},
    ...
    ]
    """
    choice_and_states = seq[:, 0]
    rews = seq[:, 1]
    # lists containing 0 for stick and 1 for switch
    # rew_commmon, rew_rare, unrew_common, unrew_rare = [], [], [], []
    out = []
    for i in range(2, len(seq)):
        choice_state_choice = choice_and_states[i-2:i+1]
        if not choice_state_choice[0] in (0, 1):  # state_choice_state
            continue
        c1, s, c2 = choice_state_choice
        assert c1 in (0, 1) and s in (2, 3) and c2 in (0, 1)
        rew = rews[i-1]
        # 1A2a1B etc. translation
        c2c_kastner_notation = f'{c1 + 1}{"a" if s == 2 else "b"}{c2 + 1}'
        if rew == 1:
            c2c_kastner_notation = c2c_kastner_notation.upper()
        data_dict = kastner_seq_to_dict(c2c_kastner_notation)[0]

        # [{'csc': (2, 'b', 2), 'stick': True, 'rew': False, 'common_transition': True}]
        out.append(data_dict)
    return out

def kastner_csc_to_classification(c1, s, c2, rew):
    assert c1 in (0, 1) and s in (2, 3) and c2 in (0, 1)
    assert rew in (0, 1)

def get_daw_bar(x):
    if x['rew']:
        if x['common_transition']:
            return 'rew_common'
        return 'rew_rare'
    else:
        if x['common_transition']:
            return 'unrew_common'
        return 'unrew_rare'


def process_episode_data(seq):
    choice_and_states = seq[:, 0]
    rews = seq[:, 1]
    # lists containing 0 for stick and 1 for switch
    # rew_commmon, rew_rare, unrew_common, unrew_rare = [], [], [], []
    out = {'rew_common': [], 'rew_rare': [], 'unrew_common': [], 'unrew_rare': []}
    for i in range(2, len(seq)):
        choice_state_choice = choice_and_states[i-2:i+1]
        if not choice_state_choice[0] in (0, 1):  # state_choice_state
            continue
        c1, s, c2 = choice_state_choice
        assert c1 in (0, 1) and s in (2, 3) and c2 in (0, 1)
        rew = rews[i-1]
        # 1A2a1B etc. translation
        c2c_kastner_notation = f'{c1 + 1}{"a" if s == 2 else "b"}{c2 + 1}'
        if rew == 1:
            c2c_kastner_notation = c2c_kastner_notation.upper()

        stick = int(c1 == c2)
        # actually we'll store not just choice but also the causing sequence
        stick = (c2c_kastner_notation, stick)
        common_transition = (c1, s) == (0, 2) or (c1, s) == (1, 3)
        if rew == 1:
            if common_transition:
                out['rew_common'].append(stick)
            else:
                out['rew_rare'].append(stick)
        else:
            if common_transition:
                out['unrew_common'].append(stick)
            else:
                out['unrew_rare'].append(stick)

    return out




def subjects_episode_data_to_daw_kastner_analysis(arr):
    """arr is a N x 400 x 2 array of subjects data (N subjects)
    """
    # list of {'rew_common': [('2B1', 0), ('1A1', 1), ...], ...}
    subjects_data = [process_episode_data(seq) for seq in arr]

    # aggregate our N subjects into one big dataset
    subjects_data_agg = {k: [] for k in subjects_data[0].keys()}
    for subject_data in subjects_data:
        for k, v in subject_data.items():
            subjects_data_agg[k].extend(v)

    # aggregate by kastner_csc
    outs = {k: {} for k in subjects_data_agg.keys()}
    for k, v in subjects_data_agg.items():
        for  (kastner_csc, stick) in v:
            if not kastner_csc in outs[k]:
                outs[k][kastner_csc] = [0, stick]
            outs[k][kastner_csc][0] += 1
            
    # convert to dataframe
    df = pd.DataFrame(columns=['daw_bar', 'stick', 'count', 'kastner_csc'])
    for daw_bar, v in outs.items():
        for kastner_csc, (count, stick) in v.items():
            df = pd.concat([df, pd.DataFrame({'daw_bar': [daw_bar], 'stick': [stick], 'count': [count], 'kastner_csc': [kastner_csc]})], ignore_index=True)
    

    #     daw_bar	stick	count	kastner_csc
    # 12	unrew_rare	1	3691	1b1
    # 13	unrew_rare	1	3475	2a2
    # 15	unrew_rare	0	3274	1b2
    # 14	unrew_rare	0	2921	2a1
    # 8	unrew_common	1	8101	1a1
    # ...
            
    return df
