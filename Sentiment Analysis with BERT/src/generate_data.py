# generate_data.py
import csv, random
POSITIVE = [
    "I loved the product, it works great and exceeded my expectations.",
    "Amazing service! Highly recommended.",
    "This is the best purchase I have made this year."
]
NEGATIVE = [
    "Terrible experience, the item broke after two days.",
    "I am very disappointed and will not buy again.",
    "Poor quality, do not recommend."
]
def generate(n=200, out='data/sentiment_data.csv'):
    rows = [['text','label']]
    for _ in range(n//2):
        rows.append([random.choice(POSITIVE), 'positive'])
    for _ in range(n//2):
        rows.append([random.choice(NEGATIVE), 'negative'])
    random.shuffle(rows[1:])
    with open(out, 'w', newline='', encoding='utf8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(rows)
    print('Wrote', out)

if __name__ == '__main__':
    generate()
