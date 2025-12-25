import json
J = json.load(open('model/run_10epochs/test_eval.json', 'r', encoding='utf-8'))
print('total', J['total'])
print('known_count', J['known_count'])
print('unknown_count', J['unknown_count'])
print('accuracy_known', J['accuracy_known'])
print('accuracy_overall', J['accuracy_overall'])
unknowns = [s for s in J.get('samples', []) if s.get('pred_idx') == -1]
print('\nUnknown sample files (up to 20):')
for s in unknowns[:20]:
    print(s['file'])
rep = J.get('classification_report_known', '')
print('\nExcerpt from classification_report_known (low recall rows):')
for line in rep.splitlines():
    parts = line.strip().split()
    if len(parts) >= 4:
        try:
            recall = float(parts[2])
            if recall < 0.9:
                print(line)
        except:
            pass
