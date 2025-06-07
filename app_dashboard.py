# app.py

from flask import Flask, jsonify, request
import pandas as pd
from collections import Counter
from datetime import datetime
import json

# --- Load dummy_data.json saat startup ---
with open('dummy_data.json', 'r') as f:
    dummy_data = json.load(f)

app = Flask(__name__)
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

@app.route('/api/analysis/label-distribution')
def label_distribution():
    labels = [item["prediction"] for item in dummy_data]
    total = len(labels)
    label_dist = dict(Counter(labels))
    for l in LABELS:
        label_dist.setdefault(l, 0)
    return jsonify({
        "total_events": total,
        "label_distribution": label_dist,
        "updated_at": max(item["timestamp"] for item in dummy_data)
    })

@app.route('/api/analysis/label-trend')
def label_trend():
    days = int(request.args.get('days', 7))
    df = pd.DataFrame(dummy_data)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    trend = {l: [] for l in LABELS}
    unique_days = sorted(df['date'].unique())[-days:]
    labels = [str(d) for d in unique_days]
    for l in LABELS:
        count_per_day = df[df['prediction'] == l].groupby('date').size().reindex(unique_days, fill_value=0).tolist()
        trend[l] = count_per_day
    return jsonify({"labels": labels, "series": trend})

@app.route('/api/analysis/label-heatmap')
def label_heatmap():
    df = pd.DataFrame(dummy_data)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    heatmap = {}
    for l in LABELS:
        counts = [0]*24
        subset = df[df['prediction'] == l]
        for h in subset['hour']:
            counts[h] += 1
        heatmap[l] = counts
    return jsonify({
        "hours": list(range(24)),
        "heatmap": heatmap
    })

@app.route('/api/analysis/label-sequence')
def label_sequence():
    sorted_data = sorted(dummy_data, key=lambda x: x["timestamp"])
    seq_counts = Counter()
    for i in range(len(sorted_data) - 1):
        from_label = sorted_data[i]['prediction']
        to_label = sorted_data[i+1]['prediction']
        seq_counts[(from_label, to_label)] += 1
    seq_list = [{"from": k[0], "to": k[1], "count": v} for k, v in seq_counts.items()]
    return jsonify({
        "sequences": seq_list,
        "total_sequences": sum(seq_counts.values())
    })

@app.route('/api/analysis/insight-summary')
def insight_summary():
    df = pd.DataFrame(dummy_data)
    top_label = df['prediction'].value_counts().idxmax()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_day = df['date'].max()
    df_last = df[df['date'] == last_day]
    counts_last = df_last['prediction'].value_counts(normalize=True)
    anomaly = None
    for l, frac in counts_last.items():
        if frac > 0.5:
            anomaly = f"Frekuensi {l} sangat dominan dalam 24 jam terakhir"
            break
    summary = f"Label terbanyak: {top_label} ({df['prediction'].value_counts()[top_label]}x)."
    if anomaly:
        summary += " Anomali: " + anomaly
    else:
        summary += " Tidak ada anomali signifikan."
    recommendation = "Perhatikan pola terbanyak dan sesuaikan aktivitas harian bayi."
    return jsonify({
        "top_label": top_label,
        "anomaly": anomaly,
        "summary": summary,
        "recommendation": recommendation
    })

@app.route('/api/predictions/history')
def prediction_history():
    return jsonify(dummy_data)

if __name__ == '__main__':
    app.run(debug=True)
