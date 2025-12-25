import json
import sys
J = json.load(open('model/run_10epochs/test_eval.json', 'r', encoding='utf-8'))
rep = J.get('per_class') or J.get('per_class')
# classification_report_known is a text block; parse lines with class stats
text = J.get('classification_report_known', '')
lines = [l for l in text.splitlines() if l.strip()]
rows = []
for l in lines:
    parts = l.strip().split()
    if len(parts) >= 4 and parts[0] not in ('accuracy', 'macro', 'weighted'):
        try:
            prec = float(parts[1])
            rec = float(parts[2])
            f1 = float(parts[3])
            supp = int(parts[4])
            rows.append((parts[0], prec, rec, f1, supp))
        except:
            pass
# print classes with recall < 0.9 or support < 30
low_recall = [r for r in rows if r[2] < 0.9]
low_support = [r for r in rows if r[4] < 30]
print('Classes with recall < 0.9:')
for name, prec, rec, f1, supp in low_recall:
    print(f"- {name}: recall={rec:.2f}, f1={f1:.2f}, support={supp}")
print('\nClasses with support < 30:')
for name, prec, rec, f1, supp in low_support:
    print(f"- {name}: support={supp}, recall={rec:.2f}, f1={f1:.2f}")
