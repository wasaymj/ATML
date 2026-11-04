* eval suites image-disjoint: True; train/eval image overlap: 0; mismatched images exclude own pair: True
* VQAv2 val filter: {'pairs_total': 95144, 'eligible': 33925, 'drop_agreement_below_8': 57047, 'drop_same_consensus': 3378, 'drop_answer_type_differs': 794}
* VQAv2 train filter: {'pairs_total': 200394, 'drop_agreement_below_8': 119896, 'drop_same_consensus': 7086, 'eligible': 71969, 'drop_answer_type_differs': 1443}
* eligible pairs by type (val / train): {'yes/no': 20963, 'other': 8039, 'number': 4923} / {'other': 17375, 'yes/no': 44033, 'number': 10561}
* grounding label rules: {'presence_min_area_frac': 0.01, 'count_min_area_frac': 0.005, 'presence_negative_shares_category': True, 'count_range': [1, 4]}; not countable: ['broccoli', 'skis']
* max tokens: {'question': 20, 'caption': 59, 'answer': 7} (caps {'question': 80, 'caption': 64, 'answer': 16})
* top answers: grounding_pool_presence: [['yes', 4000], ['no', 4000]]; grounding_pool_counting: [['1', 2582], ['2', 2337], ['3', 1837], ['4', 1244]]; grounding_dev: [['no', 12], ['yes', 12], ['2', 9], ['4', 8], ['3', 4]]; grounding_test: [['yes', 32], ['no', 32], ['1', 20], ['2', 17], ['4', 14]]; vqa_train: [['yes', 640], ['no', 640], ['1', 157], ['2', 156], ['white', 105]]; vqa_dev: [['yes', 12], ['no', 12], ['2', 5], ['1', 4], ['4', 2]]; vqa_test: [['no', 36], ['yes', 36], ['1', 18], ['2', 15], ['0', 5]]
