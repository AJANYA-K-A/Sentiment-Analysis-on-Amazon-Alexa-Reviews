import os

path = os.path.join("templates", "history.html")

content = open(path, 'r', encoding='utf-8').read()
fixed = content.replace(
    '{% if label="" ="Positive" %}',  '{% if label == "Positive" %}'
).replace(
    '{% if label="" ="Negative" %}',  '{% if label == "Negative" %}'
).replace(
    '{% if label="" ="Neutral" %}',   '{% if label == "Neutral" %}'
).replace(
    '{% if mode="" ="single" %}',     '{% if mode == "single" %}'
).replace(
    '{% if mode="" ="compare" %}',    '{% if mode == "compare" %}'
).replace(
    '{% if mode="" ="batch" %}',      '{% if mode == "batch" %}'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(fixed)

print("Fixed and saved.")

# Verify
data = open(path, encoding='utf-8').read()
if '="" =' in data:
    print("STILL BAD - manual fix needed")
else:
    print("VERIFIED CLEAN - run app.py now")